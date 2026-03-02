import os
import time
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torchao.quantization import quantize_, int8_weight_only
import numpy as np
import psutil

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "../../dataset/processed"
BASELINE_PT = "../../baseline/best_model.pt"
NUM_CLASSES = 2
BATCH_SIZE  = 32

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(path):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="weighted")
    return acc, f1

# ── Inference time ────────────────────────────────────────────────────────────
def measure_inference(model, n=100):
    sample = torch.randn(1, 3, 224, 224)
    times  = []
    with torch.no_grad():
        for _ in range(n):
            start = time.time()
            _ = model(sample)
            times.append((time.time() - start) * 1000)
    return np.mean(times), np.std(times)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_ds     = datasets.ImageFolder(DATA_DIR + "/test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading baseline model...")
    model = load_model(BASELINE_PT)

    # ── Apply Static-style Quantization via torchao ───────────────────────────
    print("Applying int8 weight-only quantization (torchao)...")
    quantize_(model, int8_weight_only())

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Evaluating...")
    acc, f1 = evaluate(model, test_loader)
    avg_time, std_time = measure_inference(model)
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    # Save and measure size
    torch.save(model.state_dict(), "model_q2.pt")
    q_size = os.path.getsize("model_q2.pt") / 1024 / 1024
    b_size = os.path.getsize(BASELINE_PT) / 1024 / 1024
    compression = b_size / q_size

    print(f"\n── Q2 Static PTQ Results (torchao int8) ──")
    print(f"Accuracy         : {acc:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Model size       : {q_size:.2f} MB  (baseline: {b_size:.2f} MB)")
    print(f"Compression ratio: {compression:.2f}x")
    print(f"Inference time   : {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"RAM usage        : {ram:.1f} MB")

    results = {
        "technique": "Q2 - Static PTQ (torchao int8)",
        "accuracy": acc,
        "f1_score": f1,
        "model_size_mb": q_size,
        "baseline_size_mb": b_size,
        "compression_ratio": compression,
        "avg_inference_ms": avg_time,
        "std_inference_ms": std_time,
        "ram_usage_mb": ram
    }
    with open("results_q2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_q2.json")

if __name__ == "__main__":
    main()