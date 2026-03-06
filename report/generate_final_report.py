import json, os, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 PageBreak, HRFlowable, Image)
from pypdf import PdfReader, PdfWriter

os.makedirs("/home/naouf/projet-embarque/report", exist_ok=True)

# ── Styles ────────────────────────────────────────────────────────
caption = ParagraphStyle('cap', fontSize=9, textColor=colors.HexColor("#595959"),
    alignment=TA_CENTER, fontName='Helvetica-Oblique', spaceAfter=16)
title_s = ParagraphStyle('ts', fontSize=14, textColor=colors.HexColor("#2E75B6"),
    fontName='Helvetica-Bold', spaceBefore=16, spaceAfter=8)

def add_plot(story, path, title, caption_text, width=16*cm, height=9*cm):
    story.append(Paragraph(title, title_s))
    story.append(Image(path, width=width, height=height))
    story.append(Spacer(1, 4))
    story.append(Paragraph(caption_text, caption))

# ── Generate training curves ──────────────────────────────────────
with open("/home/naouf/projet-embarque/baseline/results.json") as f:
    baseline = json.load(f)

history = baseline["history"]
epochs  = [h["epoch"] for h in history]
val_acc = [h["val_acc"]*100 for h in history]
val_f1  = [h["val_f1"]*100 for h in history]
loss    = [h["loss"] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(epochs, val_acc, 'b-o', linewidth=2, label='Val Accuracy', markersize=6)
ax1.plot(epochs, val_f1,  'g--s', linewidth=2, label='Val F1-Score', markersize=6)
ax1.axhline(y=97.38, color='red', linestyle=':', label='Test Accuracy finale')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Score (%)")
ax1.set_title("Courbes de Validation", fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(88, 100)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.plot(epochs, loss, 'r-o', linewidth=2, markersize=6)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.set_title("Courbe de Loss (Entrainement)", fontweight='bold')
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("/home/naouf/projet-embarque/visualizations/plot0_training.png", dpi=150, bbox_inches='tight')
plt.close()
print("Training curves generated!")

# ── Build visualisations PDF ──────────────────────────────────────
OUTPUT = "/home/naouf/projet-embarque/report/visualisations.pdf"
story  = []

story.append(Paragraph("Annexe — Visualisations et Resultats",
    ParagraphStyle('cov', fontSize=20, textColor=colors.HexColor("#2E75B6"),
    fontName='Helvetica-Bold', alignment=TA_CENTER, spaceBefore=40, spaceAfter=20)))
story.append(HRFlowable(width="100%", thickness=2,
    color=colors.HexColor("#2E75B6"), spaceAfter=20))
story.append(PageBreak())

VIZ = "/home/naouf/projet-embarque/visualizations"

add_plot(story, f"{VIZ}/plot0_training.png",
    "Figure 1 — Courbes d'Entrainement du Modele Baseline",
    "Evolution de la validation accuracy, F1-score et loss sur 10 epochs. Meilleur modele sauvegarde a l'epoch 7 (val_acc=97.15%).")
story.append(PageBreak())

add_plot(story, f"{VIZ}/plot6_confusion_matrix.png",
    "Figure 2 — Matrice de Confusion (Baseline MobileNetV2)",
    "Matrice de confusion sur le jeu de test (879 images). Sensibilite elevee pour PNEUMONIA, specificite satisfaisante pour NORMAL.",
    height=10*cm)
story.append(PageBreak())

add_plot(story, f"{VIZ}/plot1_accuracy.png",
    "Figure 3 — Precision par Technique d'Optimisation",
    "Q3, P1 et P3 surpassent le baseline (97.38%). P2 Structured Pruning souffre d'une chute significative a 72.92%.")
story.append(PageBreak())

add_plot(story, f"{VIZ}/plot2_ram.png",
    "Figure 4 — Consommation RAM par Technique",
    "Reduction de 5.5x de la RAM par rapport au baseline (2511 MB). P3 est la plus economique (458 MB). Toutes les techniques passent sous la limite VM2.")
story.append(PageBreak())

add_plot(story, f"{VIZ}/plot3_inference.png",
    "Figure 5 — Temps d'Inference par Technique",
    "P3 offre le meilleur temps d'inference (18.18 ms). Toutes les techniques restent dans des plages acceptables pour l'inference temps reel.")
story.append(PageBreak())

add_plot(story, f"{VIZ}/plot4_radar.png",
    "Figure 6 — Analyse Multi-Criteres (Radar Chart)",
    "P3 domine sur RAM et vitesse. Q3 excelle en precision. Q2 offre le meilleur equilibre global.",
    height=11*cm)
story.append(PageBreak())

add_plot(story, f"{VIZ}/plot5_heatmap.png",
    "Figure 7 — Heatmap Deploiement 3x8 (Temps d'Inference ms)",
    "VM2 est globalement la plus rapide grace a ses 2 CPUs. VM1 est plus lente car limitee a 1 CPU.",
    height=7*cm)
story.append(PageBreak())

add_plot(story, f"{VIZ}/plot7_collective.png",
    "Figure 8 — Intelligence Collective",
    "VM3 (Q2) atteint 100% sur 50 echantillons. Taux de consensus de 96% entre les 3 VMs. 2 re-interrogations pour faible confiance (<70%).")

doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm)
doc.build(story)
print("Visualisations PDF done!")

# ── Merge with main report ────────────────────────────────────────
main_pdf = "/home/naouf/projet-embarque/report/visualisations.pdf"

writer = PdfWriter()
for pdf_path in [main_pdf, OUTPUT]:
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)

final_path = "/home/naouf/projet-embarque/report/rapport_complet.pdf"
with open(final_path, "wb") as f:
    writer.write(f)

shutil.copy(final_path, "/mnt/user-data/outputs/rapport_complet.pdf")
print("Final merged report saved to rapport_complet.pdf!")
