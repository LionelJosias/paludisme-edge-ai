# 🦠 Paludisme Edge AI — Détection automatisée sur Raspberry Pi 3 B+

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![LiteRT](https://img.shields.io/badge/LiteRT-Edge%20Inference-green.svg)](https://ai.google.dev/edge/litert)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%203%20B%2B-c51a4a.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Détection automatisée des cellules et des parasites du paludisme sur des images de frottis sanguins épais avec YOLOv8n et Edge AI.**

---

## 📌 Présentation

Ce projet porte sur la conception et l'évaluation d'une solution d'**Edge AI** destinée à la détection automatisée des parasites du paludisme dans des images microscopiques de frottis sanguins épais.

L'objectif principal est de déplacer l'inférence du modèle directement sur une plateforme embarquée à ressources limitées, afin de permettre une analyse locale des images sans dépendance permanente à une infrastructure Cloud.

Le système repose sur :

- **YOLOv8n** pour la détection d'objets ;
- **TensorFlow Lite / LiteRT** pour l'exécution embarquée ;
- une optimisation **Float16** du modèle ;
- un **Raspberry Pi 3 B+** comme plateforme Edge ;
- une application Web locale basée sur **Flask** ;
- **OpenCV** pour le traitement des images ;
- **NumPy** pour les opérations numériques.

Le système permet d'importer une image de frottis sanguin depuis un navigateur Web, de réaliser localement la détection des cellules et des parasites, puis de retourner une image annotée accompagnée des résultats de détection.

---

## 🎯 Objectifs du projet

Le projet poursuit trois objectifs principaux :

1. **Établir une performance de référence** du modèle YOLOv8n pour la détection des cellules et parasites.
2. **Étudier l'effet des différentes stratégies d'optimisation** et de conversion du modèle.
3. **Évaluer la faisabilité de l'inférence Edge** sur un Raspberry Pi 3 B+ à ressources limitées.

Une attention particulière est portée au compromis entre :

- qualité de détection ;
- taille du modèle ;
- temps d'inférence ;
- consommation des ressources ;
- débit en images par seconde.

---

# 🧠 Architecture générale

```text
                         ┌─────────────────────────┐
                         │       Utilisateur       │
                         │     Navigateur Web      │
                         └────────────┬────────────┘
                                      │
                                      │ HTTP
                                      ▼
                         ┌─────────────────────────┐
                         │      Flask Server       │
                         │    Raspberry Pi 3 B+    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Prétraitement       │
                         │                         │
                         │ • Letterbox             │
                         │ • BGR → RGB             │
                         │ • Normalisation [0,1]   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      Modèle YOLOv8n     │
                         │                         │
                         │ TensorFlow Lite Float16 │
                         │        + LiteRT          │
                         │       + XNNPACK          │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Post-traitement      │
                         │                         │
                         │ • Filtrage confiance    │
                         │ • NMS                   │
                         │ • Comptage objets       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Résultats Web       │
                         │                         │
                         │ • Image annotée         │
                         │ • Cellules détectées    │
                         │ • Parasites détectés    │
                         │ • Temps de traitement   │
                         └─────────────────────────┘
paludisme-edge-ai/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── kaggle/
│   ├── 01_exploration_dataset.py
│   ├── 02_preparation_dataset.py
│   ├── 03_train_yolov8.py
│   └── 04_export_models.py
│
├── raspberry_pi/
│   ├── requirements.txt
│   │
│   ├── src/
│   │   ├── app.py
│   │   ├── detector.py
│   │   └── benchmark.py
│   │
│   └── models/
│       └── best_float16_320.tflite
│
├── docs/
│   ├── architecture.md
│   └── benchmarks.md
│
├── results/
│   ├── README.md
│   └── figures/
│
└── examples/
    ├── input/
    └── output/
