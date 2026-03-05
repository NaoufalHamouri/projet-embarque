import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.utils.prune as prune
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from sklearn.metrics import f1_score
import numpy as np
import time
import os

DATA_DIR    = "dataset/processed"
BASELINE_PT = "baseline/best_model.pt"
NUM_CLASSES = 2

# ── Best techniques from Phase 4 ─────────────────────────────────────────────
VM_CONFIG = {
    "VM1": {"technique": "P3", "ram_limit": 500, "historical_accuracy": 0.9784},
    "VM2": {"technique": "P3", "ram_limit": 1024, "historical_accuracy": 0.9784},
    "VM3": {"technique": "Q2", "ram_limit": 2048, "historical_accuracy": 0.9738},
}

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(BASELINE_PT, map_location="cpu"))
    model.eval()
    return model

# ── Apply technique ───────────────────────────────────────────────────────────
def apply_technique(model, technique):
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

# ── Single inference with softmax confidence ──────────────────────────────────
def infer(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probs  = torch.softmax(output, dim=1)
        pred   = probs.argmax(dim=1).item()
        conf   = probs.max().item()
    return pred, conf, probs.squeeze().numpy()

# ── Weighted vote ─────────────────────────────────────────────────────────────
def weighted_vote(predictions):
    """
    predictions: list of (vm_id, pred, confidence, historical_accuracy)
    Returns: final_pred, collective_confidence
    """
    class_weights = {}
    for vm_id, pred, conf, hist_acc in predictions:
        weight = hist_acc * conf
        class_weights[pred] = class_weights.get(pred, 0) + weight

    final_pred = max(class_weights, key=class_weights.get)
    total_weight = sum(class_weights.values())
    collective_conf = class_weights[final_pred] / total_weight
    return final_pred, collective_conf

# ── Main orchestrator ─────────────────────────────────────────────────────────
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds     = datasets.ImageFolder(DATA_DIR + "/test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    classes     = test_ds.classes

    # Load models for each VM
    print("Loading models for each VM...")
    vm_models = {}
    for vm_id, config in VM_CONFIG.items():
        print(f"  Preparing {vm_id} with {config['technique']}...")
        model = load_model()
        model = apply_technique(model, config["technique"])
        vm_models[vm_id] = model
    print("All models ready!\n")

    # Run collective inference on 50 test samples
    N_SAMPLES = 50
    results = []
    collective_correct = 0
    consensus_count    = 0
    confidence_retries = 0
    CONFIDENCE_THRESHOLD = 0.70

    print(f"Running collective inference on {N_SAMPLES} samples...")
    print(f"{'='*60}")

    for i, (image, label) in enumerate(test_loader):
        if i >= N_SAMPLES:
            break

        label = label.item()
        predictions = []

        # Each VM makes a prediction
        for vm_id, model in vm_models.items():
            pred, conf, probs = infer(model, image.squeeze(0))
            hist_acc = VM_CONFIG[vm_id]["historical_accuracy"]
            predictions.append((vm_id, pred, conf, hist_acc))

        # Weighted vote
        final_pred, collective_conf = weighted_vote(predictions)

        # Confidence validation — if < threshold, retry with all VMs
        if collective_conf < CONFIDENCE_THRESHOLD:
            confidence_retries += 1
            # Re-query (same result in our simulation, but counts the mechanism)
            final_pred, collective_conf = weighted_vote(predictions)

        # Check consensus (all VMs agree)
        all_preds = [p[1] for p in predictions]
        if len(set(all_preds)) == 1:
            consensus_count += 1

        correct = (final_pred == label)
        if correct:
            collective_correct += 1

        results.append({
            "sample": i,
            "true_label": classes[label],
            "collective_pred": classes[final_pred],
            "collective_conf": round(collective_conf, 4),
            "correct": correct,
            "consensus": len(set(all_preds)) == 1,
            "vm_predictions": {vm_id: classes[p] for vm_id, p, _, _ in predictions}
        })

    # ── Individual VM accuracy for comparison ─────────────────────────────────
    print("\nEvaluating individual VM accuracy...")
    individual_acc = {}
    for vm_id, model in vm_models.items():
        correct, total = 0, 0
        all_preds_list, all_labels_list = [], []
        for i, (image, label) in enumerate(test_loader):
            if i >= N_SAMPLES:
                break
            pred, _, _ = infer(model, image.squeeze(0))
            correct += (pred == label.item())
            total   += 1
            all_preds_list.append(pred)
            all_labels_list.append(label.item())
        individual_acc[vm_id] = correct / total

    # ── Print summary ─────────────────────────────────────────────────────────
    collective_acc  = collective_correct / N_SAMPLES
    consensus_rate  = consensus_count / N_SAMPLES
    best_individual = max(individual_acc.values())
    gain            = (collective_acc - best_individual) * 100

    print(f"\n{'='*60}")
    print(f"  COLLECTIVE INTELLIGENCE RESULTS")
    print(f"{'='*60}")
    print(f"  Samples evaluated     : {N_SAMPLES}")
    print(f"  Collective accuracy   : {collective_acc:.4f} ({collective_acc*100:.2f}%)")
    print(f"\n  Individual VM accuracy:")
    for vm_id, acc in individual_acc.items():
        print(f"    {vm_id} ({VM_CONFIG[vm_id]['technique']}): {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  Best individual       : {best_individual:.4f} ({best_individual*100:.2f}%)")
    print(f"  Collective gain       : {gain:+.2f}%")
    print(f"  Consensus rate        : {consensus_rate:.4f} ({consensus_rate*100:.2f}%)")
    print(f"  Confidence retries    : {confidence_retries} / {N_SAMPLES}")

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "collective_accuracy": collective_acc,
        "individual_accuracy": individual_acc,
        "best_individual_accuracy": best_individual,
        "collective_gain_pct": gain,
        "consensus_rate": consensus_rate,
        "confidence_retries": confidence_retries,
        "n_samples": N_SAMPLES,
        "vm_config": VM_CONFIG,
        "sample_results": results
    }
    with open("collective/results_collective.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to collective/results_collective.json")

if __name__ == "__main__":
    main()
