import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from torch.utils.data import DataLoader
import paho.mqtt.client as mqtt
import numpy as np
import psutil
import os

# ── ThingsBoard MQTT Config ───────────────────────────────────────────────────
TB_HOST  = "localhost"
TB_PORT  = 1883
TOPIC    = "v1/devices/me/telemetry"

VM_TOKENS = {
    "VM1": "1315upeoppeeja2o8y6d",
    "VM2": "jw066736s50vd1r0trb0",
    "VM3": "0urqj44d5qulp6jj8z0s",
}

VM_TECHNIQUES = {
    "VM1": "P3",
    "VM2": "P3",
    "VM3": "Q2",
}

DATA_DIR    = "../dataset/processed"
BASELINE_PT = "../baseline/best_model.pt"
NUM_CLASSES = 2
CLASSES     = ["NORMAL", "PNEUMONIA"]

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(technique):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(BASELINE_PT, map_location="cpu"))
    model.eval()

    if technique == "P3":
        params = [(m, "weight") for _, m in model.named_modules()
                  if isinstance(m, (nn.Linear, nn.Conv2d))]
        for _ in range(3):
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.1)
        for m, p in params:
            prune.remove(m, p)
    elif technique == "Q2":
        quantize_(model, Int8WeightOnlyConfig())

    model.eval()
    return model

# ── Single inference ──────────────────────────────────────────────────────────
def run_inference(model, image):
    start = time.time()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs  = torch.softmax(output, dim=1)
        pred   = probs.argmax(dim=1).item()
        conf   = probs.max().item()
    elapsed_ms = (time.time() - start) * 1000
    return pred, conf, elapsed_ms

# ── Send telemetry to ThingsBoard ─────────────────────────────────────────────
def send_telemetry(vm_id, token, payload):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(token)
    client.connect(TB_HOST, TB_PORT, 60)
    client.loop_start()
    result = client.publish(TOPIC, json.dumps(payload))
    time.sleep(0.5)
    client.loop_stop()
    client.disconnect()
    return result.rc == 0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds     = datasets.ImageFolder(DATA_DIR + "/test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)

    print("Loading models...")
    models = {vm_id: load_model(tech) for vm_id, tech in VM_TECHNIQUES.items()}
    print("All models ready!\n")

    print("Starting telemetry stream to ThingsBoard...")
    print(f"{'='*60}")

    patient_id = 1000
    for i, (image, label) in enumerate(test_loader):
        if i >= 20:
            break

        true_label = CLASSES[label.item()]
        image = image.squeeze(0)

        for vm_id, model in models.items():
            pred, conf, infer_ms = run_inference(model, image)
            ram_mb  = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            cpu_pct = psutil.cpu_percent(interval=0.1)

            payload = {
                "vm_id":             vm_id,
                "technique":         VM_TECHNIQUES[vm_id],
                "prediction":        CLASSES[pred],
                "true_label":        true_label,
                "correct":           CLASSES[pred] == true_label,
                "confidence":        round(conf, 4),
                "inference_time_ms": round(infer_ms, 2),
                "cpu_usage_pct":     round(cpu_pct, 1),
                "ram_usage_mb":      round(ram_mb, 1),
                "patient_id":        f"P-{patient_id:04d}",
            }

            success = send_telemetry(vm_id, VM_TOKENS[vm_id], payload)
            status  = "✓" if success else "✗"
            print(f"[{status}] {vm_id} | Patient P-{patient_id:04d} | "
                  f"Pred: {CLASSES[pred]:9s} | True: {true_label:9s} | "
                  f"Conf: {conf:.2f} | {infer_ms:.1f}ms | RAM: {ram_mb:.0f}MB")

        patient_id += 1
        time.sleep(1)  # 1 second between patients

    print(f"\nDone! Check your ThingsBoard dashboard at http://localhost:8080")

if __name__ == "__main__":
    main()
