import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import os

os.makedirs("visualizations", exist_ok=True)

# ── Load all results ──────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)

baseline = load("baseline/results.json")
q1 = load("optimization/Q1_dynamic_quant/results_q1.json")
q2 = load("optimization/Q2_static_ptq/results_q2.json")
q3 = load("optimization/Q3_qat/results_q3.json")
q4 = load("optimization/Q4_weight_only/results_q4.json")
q5 = load("optimization/Q5_mixed_precision/results_q5.json")
p1 = load("optimization/P1_unstructured/results_p1.json")
p2 = load("optimization/P2_structured/results_p2.json")
p3 = load("optimization/P3_magnitude/results_p3.json")
vm1 = load("deployment/results_VM1.json")
vm2 = load("deployment/results_VM2.json")
vm3 = load("deployment/results_VM3.json")
collective = load("collective/results_collective.json")

techniques = ["Baseline", "Q1", "Q2", "Q3", "Q4", "Q5", "P1", "P2", "P3"]
results = [baseline, q1, q2, q3, q4, q5, p1, p2, p3]
colors_list = ["#95A5A6","#2E86AB","#2E86AB","#2E86AB","#2E86AB","#2E86AB","#E67E22","#E67E22","#E67E22"]

def get_acc(r):
    return r.get("accuracy", r.get("test_accuracy", 0)) * 100

def get_ram(r):
    return r.get("ram_usage_mb", 0)

def get_infer(r):
    return r.get("avg_inference_ms", 0)

acc   = [get_acc(r) for r in results]
ram   = [get_ram(r) for r in results]
infer = [get_infer(r) for r in results]

# ══════════════════════════════════════════════════════
# PLOT 1 — Accuracy Comparison Bar Chart
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(techniques, acc, color=colors_list, edgecolor='white', linewidth=1.5, zorder=3)
ax.set_ylim(60, 101)
ax.axhline(y=97.38, color='red', linestyle='--', linewidth=1.5, label='Baseline 97.38%', zorder=2)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Precision par Technique d'Optimisation", fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
blue_patch = mpatches.Patch(color='#2E86AB', label='Quantification')
orange_patch = mpatches.Patch(color='#E67E22', label='Elagage')
gray_patch = mpatches.Patch(color='#95A5A6', label='Baseline')
ax.legend(handles=[gray_patch, blue_patch, orange_patch], loc='lower left')
plt.tight_layout()
plt.savefig("visualizations/plot1_accuracy.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 1 done")

# ══════════════════════════════════════════════════════
# PLOT 2 — RAM Usage Comparison
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(techniques, ram, color=colors_list, edgecolor='white', linewidth=1.5, zorder=3)
ax.axhline(y=500, color='red', linestyle='--', linewidth=1.5, label='Limite VM1 (500 MB)', zorder=2)
ax.axhline(y=1024, color='orange', linestyle='--', linewidth=1.5, label='Limite VM2 (1024 MB)', zorder=2)
ax.set_ylabel("RAM (MB)", fontsize=12)
ax.set_title("Consommation RAM par Technique d'Optimisation", fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, ram):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("visualizations/plot2_ram.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 done")

# ══════════════════════════════════════════════════════
# PLOT 3 — Inference Time Comparison
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(techniques, infer, color=colors_list, edgecolor='white', linewidth=1.5, zorder=3)
ax.set_ylabel("Temps d'inference (ms)", fontsize=12)
ax.set_title("Temps d'Inference par Technique d'Optimisation", fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, infer):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("visualizations/plot3_inference.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 done")

# ══════════════════════════════════════════════════════
# PLOT 4 — Radar Chart
# ══════════════════════════════════════════════════════
selected = ["Q1","Q2","Q3","P1","P3"]
sel_results = [q1, q2, q3, p1, p3]
categories = ["Precision", "RAM\n(inverse)", "Vitesse\n(inverse)", "F1-Score"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, size=11)
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25","0.5","0.75","1.0"], color="grey", size=8)
plt.ylim(0, 1)

colors_radar = ['#2E86AB','#A23B72','#F18F01','#C73E1D','#3B1F2B']
for i, (tech, res) in enumerate(zip(selected, sel_results)):
    vals = [
        res.get("accuracy", res.get("test_accuracy", 0)),
        1 - (res.get("ram_usage_mb", 0) - 400) / 2800,
        1 - (res.get("avg_inference_ms", 0) - 18) / 30,
        res.get("f1_score", res.get("test_f1", 0))
    ]
    vals = [max(0, min(1, v)) for v in vals]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, linestyle='solid', label=tech, color=colors_radar[i])
    ax.fill(angles, vals, alpha=0.1, color=colors_radar[i])

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_title("Comparaison Multi-Criteres des Meilleures Techniques", size=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("visualizations/plot4_radar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 4 done")

# ══════════════════════════════════════════════════════
# PLOT 5 — VM Deployment Heatmap
# ══════════════════════════════════════════════════════
tech_ids = ["Q1","Q2","Q3","Q4","Q5","P1","P2","P3"]
vm_data = {
    "VM1": [vm1["results"][t]["avg_inference_ms"] if isinstance(vm1["results"][t], dict) else 0 for t in tech_ids],
    "VM2": [vm2["results"][t]["avg_inference_ms"] if isinstance(vm2["results"][t], dict) else 0 for t in tech_ids],
    "VM3": [vm3["results"][t]["avg_inference_ms"] if isinstance(vm3["results"][t], dict) else 0 for t in tech_ids],
}
matrix = np.array([vm_data["VM1"], vm_data["VM2"], vm_data["VM3"]])

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(tech_ids)))
ax.set_xticklabels(tech_ids, fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(["VM1 (500MB)", "VM2 (1GB)", "VM3 (2GB)"], fontsize=11)
ax.set_title("Heatmap — Temps d'Inference (ms) par VM et Technique", fontsize=13, fontweight='bold', pad=15)
for i in range(3):
    for j in range(len(tech_ids)):
        ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if matrix[i,j] > 80 else 'black')
plt.colorbar(im, ax=ax, label="ms")
plt.tight_layout()
plt.savefig("visualizations/plot5_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 5 done")

# ══════════════════════════════════════════════════════
# PLOT 6 — Confusion Matrix
# ══════════════════════════════════════════════════════
print("Generating confusion matrix (loading model)...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_ds = datasets.ImageFolder("dataset/processed/test", transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("baseline/best_model.pt", map_location="cpu"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

cm_matrix = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=["NORMAL", "PNEUMONIA"])
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title("Matrice de Confusion — Modele Baseline MobileNetV2\n(Test Set: 879 images)",
             fontsize=13, fontweight='bold', pad=15)
tn, fp, fn, tp = cm_matrix.ravel()
ax.text(0.5, -0.12,
        f"Sensibilite: {tp/(tp+fn)*100:.1f}%  |  Specificite: {tn/(tn+fp)*100:.1f}%  |  Precision: {(tp+tn)/(tp+tn+fp+fn)*100:.1f}%",
        transform=ax.transAxes, ha='center', fontsize=10, color='#1F4E79')
plt.tight_layout()
plt.savefig("visualizations/plot6_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 6 done")

# ══════════════════════════════════════════════════════
# PLOT 7 — Collective Intelligence
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vms = ["VM1\n(P3)", "VM2\n(P3)", "VM3\n(Q2)", "Collectif"]
acc_vals = [
    collective["individual_accuracy"]["VM1"] * 100,
    collective["individual_accuracy"]["VM2"] * 100,
    collective["individual_accuracy"]["VM3"] * 100,
    collective["collective_accuracy"] * 100,
]
bar_colors = ['#E67E22', '#E67E22', '#2E86AB', '#1F4E79']
bars = axes[0].bar(vms, acc_vals, color=bar_colors, edgecolor='white', linewidth=1.5, zorder=3)
axes[0].set_ylim(90, 102)
axes[0].set_ylabel("Precision (%)", fontsize=11)
axes[0].set_title("Precision Individuelle vs Collective\n(50 echantillons)", fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, zorder=1)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
for bar, val in zip(bars, acc_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

consensus = collective["consensus_rate"] * 100
labels = [f'Consensus\n({consensus:.0f}%)', f'Desaccord\n({100-consensus:.0f}%)']
sizes = [consensus, 100-consensus]
axes[1].pie(sizes, labels=labels, colors=['#2E86AB', '#E74C3C'],
            autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
axes[1].set_title("Taux de Consensus\nentre les 3 VMs", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("visualizations/plot7_collective.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 7 done")

print("\nAll 7 plots saved to visualizations/")