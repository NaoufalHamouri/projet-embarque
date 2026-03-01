import os
import time
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import numpy as np
import psutil

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "../dataset/processed"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2

print(f"Using device: {DEVICE}")

# ── Transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Datasets ────────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(DATA_DIR + "/train", transform=train_transform)
val_ds   = datasets.ImageFolder(DATA_DIR + "/val",   transform=val_test_transform)
test_ds  = datasets.ImageFolder(DATA_DIR + "/test",  transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Classes: {train_ds.classes}")

# ── Class weights (handle imbalance) ────────────────────────────────────────
counts     = np.array([len(os.listdir(f"{DATA_DIR}/train/{c}")) for c in train_ds.classes])
weights    = 1.0 / counts
weights    = torch.tensor(weights / weights.sum(), dtype=torch.float).to(DEVICE)

# ── Model ────────────────────────────────────────────────────────────────────
model = mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ── Training ─────────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="weighted")
    return acc, f1

best_val_acc = 0
history = []

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    start = time.time()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    val_acc, val_f1 = evaluate(val_loader)
    elapsed = time.time() - start

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} "
          f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {elapsed:.1f}s")

    history.append({"epoch": epoch+1, "loss": running_loss/len(train_loader),
                    "val_acc": val_acc, "val_f1": val_f1})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

# ── Final Evaluation on Test Set ─────────────────────────────────────────────
print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load("best_model.pt"))
test_acc, test_f1 = evaluate(test_loader)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test F1 Score : {test_f1:.4f}")

# ── Inference time & RAM ──────────────────────────────────────────────────────
print("\nMeasuring inference time (100 images)...")
model.eval()
sample_images = torch.randn(100, 3, 224, 224).to(DEVICE)
times = []
for i in range(100):
    img = sample_images[i].unsqueeze(0)
    start = time.time()
    with torch.no_grad():
        _ = model(img)
    times.append((time.time() - start) * 1000)

avg_time = np.mean(times)
ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print(f"Avg inference time : {avg_time:.2f} ms")
print(f"RAM usage          : {ram_usage:.1f} MB")

# ── Model size ────────────────────────────────────────────────────────────────
model_size = os.path.getsize("best_model.pt") / 1024 / 1024
print(f"Model size         : {model_size:.2f} MB")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "test_accuracy": test_acc,
    "test_f1": test_f1,
    "avg_inference_ms": avg_time,
    "ram_usage_mb": ram_usage,
    "model_size_mb": model_size,
    "history": history
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to baseline/results.json")
print("Training complete!")