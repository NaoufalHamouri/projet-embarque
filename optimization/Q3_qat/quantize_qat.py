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
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
import numpy as np
import psutil

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "../../dataset/processed"
BASELINE_PT = "../../baseline/best_model.pt"
NUM_CLASSES = 2
BATCH_SIZE  = 32
EPOCHS      = 3   # Fine-tuning epochs (less than baseline)
LR          = 1e-4
DEVICE      = torch.device("cpu")

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(path):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
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
    model.eval()
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
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds    = datasets.ImageFolder(DATA_DIR + "/train", transform=transform_train)
    test_ds     = datasets.ImageFolder(DATA_DIR + "/test",  transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Load baseline ─────────────────────────────────────────────────────────
    print("Loading baseline model...")
    model = load_model(BASELINE_PT)

    # ── Apply QAT ─────────────────────────────────────────────────────────────
    print("Preparing QAT quantizer...")
    quantizer = Int8DynActInt4WeightQATQuantizer()
    model = quantizer.prepare(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── Fine-tune with QAT ────────────────────────────────────────────────────
    print(f"Fine-tuning with QAT for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        start = time.time()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Time: {elapsed:.1f}s")

    # ── Convert to quantized model ────────────────────────────────────────────
    print("Converting QAT model to quantized...")
    model = quantizer.convert(model)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Evaluating...")
    acc, f1 = evaluate(model, test_loader)
    avg_time, std_time = measure_inference(model)
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    torch.save(model.state_dict(), "model_q3.pt")
    q_size = os.path.getsize("model_q3.pt") / 1024 / 1024
    b_size = os.path.getsize(BASELINE_PT) / 1024 / 1024
    compression = b_size / q_size

    print(f"\n── Q3 QAT Results ──")
    print(f"Accuracy         : {acc:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Model size       : {q_size:.2f} MB  (baseline: {b_size:.2f} MB)")
    print(f"Compression ratio: {compression:.2f}x")
    print(f"Inference time   : {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"RAM usage        : {ram:.1f} MB")

    results = {
        "technique": "Q3 - QAT (Int8DynActInt4Weight)",
        "accuracy": acc,
        "f1_score": f1,
        "model_size_mb": q_size,
        "baseline_size_mb": b_size,
        "compression_ratio": compression,
        "avg_inference_ms": avg_time,
        "std_inference_ms": std_time,
        "ram_usage_mb": ram
    }
    with open("results_q3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_q3.json")

if __name__ == "__main__":
    main()