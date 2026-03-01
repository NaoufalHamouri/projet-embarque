# Projet : Inférence médicale embarquée

Quantification et déploiement de modèles de deep learning
sur des machines virtuelles à ressources contraintes.

## Module
Systèmes embarqués et objets connectés – Master Data Science – ENS Martil

## Dataset
NIH Chest X-ray (Pneumonia Detection)

## Stack
- Python 3 / PyTorch
- Docker (VM simulation)
- ThingsBoard (IoT monitoring)

## Structure
- `dataset/` — preprocessing scripts
- `baseline/` — baseline model training
- `optimization/` — 8 optimization techniques (Q1-Q5, P1-P3)
- `deployment/` — deployment scripts per VM
- `collective/` — orchestrator and voting mechanism
- `thingsboard/` — MQTT client and dashboards
- `results/` — CSV tables and figures
