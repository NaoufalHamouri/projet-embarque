import json
import numpy as np

# ── Load results from all 3 VMs ───────────────────────────────────────────────
def load_results(vm_id):
    with open(f"deployment/results_{vm_id}.json") as f:
        return json.load(f)

vm1 = load_results("VM1")["results"]
vm2 = load_results("VM2")["results"]
vm3 = load_results("VM3")["results"]

techniques = ["Q1", "Q2", "Q3", "Q4", "Q5", "P1", "P2", "P3"]

# ── Normalize a metric (lower is better → invert) ────────────────────────────
def normalize(values, lower_is_better=False):
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.ones_like(arr)
    norm = (arr - mn) / (mx - mn)
    return 1 - norm if lower_is_better else norm

# ── Build score for each VM ───────────────────────────────────────────────────
def compute_scores(vm_results, weights):
    """
    weights = dict with keys: accuracy, ram, speed, cpu
    each value is the weight for that metric
    """
    techs = [t for t in techniques if isinstance(vm_results[t], dict)]
    
    acc_raw   = [vm_results[t]["accuracy"]         for t in techs]
    ram_raw   = [vm_results[t]["ram_mb"]            for t in techs]
    speed_raw = [vm_results[t]["avg_inference_ms"]  for t in techs]
    cpu_raw   = [vm_results[t]["cpu_pct"]           for t in techs]

    acc_norm   = normalize(acc_raw,   lower_is_better=False)
    ram_norm   = normalize(ram_raw,   lower_is_better=True)   # less RAM = better
    speed_norm = normalize(speed_raw, lower_is_better=True)   # less time = better
    cpu_norm   = normalize(cpu_raw,   lower_is_better=True)   # less CPU = better

    scores = {}
    for i, t in enumerate(techs):
        score = (
            weights.get("accuracy", 0) * acc_norm[i] +
            weights.get("ram",      0) * ram_norm[i] +
            weights.get("speed",    0) * speed_norm[i] +
            weights.get("cpu",      0) * cpu_norm[i]
        )
        scores[t] = round(score, 4)
    return scores

# ── VM1: RAM(0.4) + CPU(0.4) + Accuracy(0.2) ─────────────────────────────────
vm1_scores = compute_scores(vm1, {"ram": 0.4, "cpu": 0.4, "accuracy": 0.2})

# ── VM2: RAM(0.3) + Speed(0.3) + Accuracy(0.4) ───────────────────────────────
vm2_scores = compute_scores(vm2, {"ram": 0.3, "speed": 0.3, "accuracy": 0.4})

# ── VM3: Accuracy(0.6) + Speed(0.25) + RAM(0.15) ─────────────────────────────
vm3_scores = compute_scores(vm3, {"accuracy": 0.6, "speed": 0.25, "ram": 0.15})

# ── Print results ─────────────────────────────────────────────────────────────
def print_scores(vm_name, scores, vm_results):
    print(f"\n{'='*55}")
    print(f"  {vm_name} — Technique Ranking")
    print(f"{'='*55}")
    print(f"{'Rank':<6}{'Tech':<6}{'Score':<8}{'Accuracy':<12}{'RAM MB':<10}{'Infer ms':<12}{'CPU%'}")
    print("-"*55)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (tech, score) in enumerate(ranked, 1):
        r = vm_results[tech]
        if isinstance(r, dict):
            print(f"{rank:<6}{tech:<6}{score:<8.4f}{r['accuracy']:<12.4f}{r['ram_mb']:<10.1f}{r['avg_inference_ms']:<12.2f}{r['cpu_pct']}")
    best = ranked[0][0]
    print(f"\n  ★ BEST TECHNIQUE: {best} (score={ranked[0][1]})")
    return best

best_vm1 = print_scores("VM1 (IoT Sensor — RAM+CPU priority)", vm1_scores, vm1)
best_vm2 = print_scores("VM2 (IoT Gateway — balanced)", vm2_scores, vm2)
best_vm3 = print_scores("VM3 (Edge Server — accuracy priority)", vm3_scores, vm3)

# ── Justifications ────────────────────────────────────────────────────────────
justifications = {
    "VM1": f"""VM1 has only 500MB RAM and 1 CPU core, making resource efficiency critical.
The scoring heavily weights RAM (0.4) and CPU usage (0.4) with only 0.2 for accuracy.
{best_vm1} achieves the best balance of low RAM consumption and minimal CPU usage,
making it the most viable technique for this severely constrained IoT sensor device.""",

    "VM2": f"""VM2 represents a mid-tier IoT gateway with 1GB RAM and 2 CPU cores.
The balanced scoring (RAM 0.3, Speed 0.3, Accuracy 0.4) reflects its role as
an intermediate node that must maintain reasonable accuracy while processing
requests efficiently. {best_vm2} provides the best overall balance for this profile.""",

    "VM3": f"""VM3 is the most capable device (2GB RAM, 2 CPUs) acting as an edge server.
Accuracy is the top priority (0.6 weight) as this device makes final diagnostic decisions.
Speed (0.25) and RAM (0.15) are secondary concerns. {best_vm3} delivers the highest
accuracy with acceptable speed, making it ideal for this edge server role."""
}

print(f"\n{'='*55}")
print("  JUSTIFICATIONS")
print(f"{'='*55}")
for vm, text in justifications.items():
    print(f"\n{vm}:")
    print(text)

# ── Save selection results ────────────────────────────────────────────────────
selection = {
    "VM1": {"best_technique": best_vm1, "scores": vm1_scores},
    "VM2": {"best_technique": best_vm2, "scores": vm2_scores},
    "VM3": {"best_technique": best_vm3, "scores": vm3_scores},
    "justifications": justifications
}
with open("deployment/phase4_selection.json", "w") as f:
    json.dump(selection, f, indent=2)

print(f"\n\nResults saved to deployment/phase4_selection.json")
print(f"\nSUMMARY:")
print(f"  VM1 best technique: {best_vm1}")
print(f"  VM2 best technique: {best_vm2}")
print(f"  VM3 best technique: {best_vm3}")
