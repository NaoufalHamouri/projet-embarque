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
import numpy as np
import psutil

DATA_DIR    = "/app/dataset/processed"
BASELINE_PT = "/app/baseline/best_model.pt"
NUM_CLASSES = 2
BATCH_SIZE  = 32
VM_ID       = os.environ.get("VM_ID", "VM1")
VM_RAM      = int(os.environ.get("VM_RAM", 500))

def load_model(path):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
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
    return correct/total, f1_score(all_labels, all_preds, average="weighted")

def measure_inference(model, n=10):
    sample = torch.randn(1, 3, 224, 224)
    times = []
    with torch.no_grad():
        for _ in range(n):
            start = time.time()
            _ = model(sample)
            times.append((time.time() - start) * 1000)
    return np.mean(times), np.std(times)

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def get_cpu():
    return psutil.cpu_percent(interval=1)

def apply_q1(model):
    return torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

def apply_q2(model):
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    quantize_(model, Int8WeightOnlyConfig())
    return model

def apply_q4(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.data
            scale = w.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / 127.0
            module.weight.data = ((w/scale).round().clamp(-128,127) * scale).float()
    return model

def apply_q5(model):
    import numpy as np
    sensitivity = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            sensitivity[name] = module.weight.data.var().item()
    vals = list(sensitivity.values())
    threshold = np.percentile(vals, 70)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and name in sensitivity:
            w = module.weight.data
            if sensitivity[name] >= threshold:
                scale = w.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / 127.0
            else:
                scale = w.abs().max().clamp(min=1e-8) / 127.0
            module.weight.data = ((w/scale).round().clamp(-128,127) * scale).float()
    return model

def apply_p1(model):
    import torch.nn.utils.prune as prune
    params = [(m, "weight") for _, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.3)
    for m, p in params: prune.remove(m, p)
    return model

def apply_p2(model):
    import torch.nn.utils.prune as prune
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.weight.shape[0] > 1:
            prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)
            prune.remove(module, "weight")
    return model

def apply_p3(model):
    import torch.nn.utils.prune as prune
    params = [(m, "weight") for _, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    for _ in range(3):
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.1)
    for m, p in params: prune.remove(m, p)
    return model

TECHNIQUES = {
    "Q1": apply_q1,
    "Q2": apply_q2,
    "Q3": None,   # QAT needs pretrained - use Q1 result
    "Q4": apply_q4,
    "Q5": apply_q5,
    "P1": apply_p1,
    "P2": apply_p2,
    "P3": apply_p3,
}

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds     = datasets.ImageFolder(DATA_DIR + "/test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}
    print(f"\n{'='*50}")
    print(f"Running on {VM_ID} (RAM limit: {VM_RAM} MB)")
    print(f"{'='*50}")

    for tech_id, apply_fn in TECHNIQUES.items():
        print(f"\n--- {tech_id} ---")
        try:
            model = load_model(BASELINE_PT)

            # Check RAM before applying
            ram_before = get_ram()
            if ram_before > VM_RAM * 0.85:
                print(f"  OOM risk — skipping (RAM already at {ram_before:.0f} MB)")
                results[tech_id] = "OOM"
                continue

            if apply_fn is None:
                # Q3: use Q1 dynamic quantization as proxy (Q3 weights incompatible)
                model = apply_q1(model)
                model.eval()
            else:
                model = apply_fn(model)

            model.eval()
            acc, f1 = evaluate(model, test_loader)
            avg_time, std_time = measure_inference(model, n=10)
            ram = get_ram()
            cpu = get_cpu()

            print(f"  Accuracy : {acc:.4f}")
            print(f"  Infer    : {avg_time:.2f} ms")
            print(f"  RAM      : {ram:.1f} MB")
            print(f"  CPU      : {cpu:.1f}%")

            results[tech_id] = {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "avg_inference_ms": round(avg_time, 2),
                "std_inference_ms": round(std_time, 2),
                "ram_mb": round(ram, 1),
                "cpu_pct": round(cpu, 1)
            }

        except MemoryError:
            print(f"  OOM — out of memory")
            results[tech_id] = "OOM"
        except Exception as e:
            print(f"  ERROR: {e}")
            results[tech_id] = f"ERROR: {str(e)}"

    # Save results
    out_path = f"/app/deployment/results_{VM_ID}.json"
    with open(out_path, "w") as f:
        json.dump({"vm_id": VM_ID, "vm_ram_mb": VM_RAM, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
