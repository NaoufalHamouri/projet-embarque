import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import psutil

DATA_DIR    = "../../dataset/processed"
BASELINE_PT = "../../baseline/best_model.pt"
NUM_CLASSES = 2
BATCH_SIZE  = 32

def load_model(path):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def apply_magnitude_pruning(model, iterations=3, amount_per_iter=0.1):
    """Iterative magnitude pruning — prune 10% per iteration x 3 = ~27% total"""
    params_to_prune = [
        (module, "weight")
        for _, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]
    for i in range(iterations):
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount_per_iter,
        )
        total = sum(m.weight.nelement() for m, _ in params_to_prune)
        zeros = sum((m.weight == 0).sum().item() for m, _ in params_to_prune)
        print(f"  Iteration {i+1}: sparsity = {100*zeros/total:.1f}%")

    # Make pruning permanent
    for module, param in params_to_prune:
        prune.remove(module, param)

    return model

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
    return correct / total, f1_score(all_labels, all_preds, average="weighted")

def measure_inference(model, n=100):
    sample = torch.randn(1, 3, 224, 224)
    times = []
    with torch.no_grad():
        for _ in range(n):
            start = time.time()
            _ = model(sample)
            times.append((time.time() - start) * 1000)
    return np.mean(times), np.std(times)

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds     = datasets.ImageFolder(DATA_DIR + "/test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Loading baseline model...")
    model = load_model(BASELINE_PT)

    print("Applying iterative magnitude pruning (3 x 10%)...")
    model = apply_magnitude_pruning(model, iterations=3, amount_per_iter=0.1)

    print("Evaluating...")
    acc, f1 = evaluate(model, test_loader)
    avg_time, std_time = measure_inference(model)
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    torch.save(model.state_dict(), "model_p3.pt")
    p_size = os.path.getsize("model_p3.pt") / 1024 / 1024
    b_size = os.path.getsize(BASELINE_PT) / 1024 / 1024

    print(f"\n── P3 Magnitude Pruning Results ──")
    print(f"Accuracy         : {acc:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Model size       : {p_size:.2f} MB  (baseline: {b_size:.2f} MB)")
    print(f"Compression ratio: {b_size/p_size:.2f}x")
    print(f"Inference time   : {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"RAM usage        : {ram:.1f} MB")

    results = {
        "technique": "P3 - Magnitude Pruning (3x10% iterative)",
        "accuracy": acc, "f1_score": f1,
        "model_size_mb": p_size, "baseline_size_mb": b_size,
        "compression_ratio": b_size/p_size,
        "avg_inference_ms": avg_time, "std_inference_ms": std_time,
        "ram_usage_mb": ram
    }
    with open("results_p3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_p3.json")

if __name__ == "__main__":
    main()
