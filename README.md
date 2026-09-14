# 🦟 Paludisme Edge AI — Détection embarquée des parasites du paludisme

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLOv26n-Ultralytics-orange.svg)](https://docs.ultralytics.com/)
[![Edge](https://img.shields.io/badge/Edge-Raspberry%20Pi%203%20B%2B-red.svg)]
[![Runtime](https://img.shields.io/badge/Runtime-LiteRT%20%2F%20TFLite-green.svg)]

Projet académique consacré à la **détection automatisée des parasites du paludisme dans des images de frottis sanguins épais**, en combinant l'intelligence artificielle, les modèles YOLO légers et l'Edge Computing.

> **Statut : preuve expérimentale de faisabilité.** Ce projet ne constitue pas un dispositif médical ni un outil de diagnostic clinique validé.

---

# 1. Présentation du projet

Le projet couvre l'ensemble de la chaîne de traitement, depuis la préparation des données jusqu'à l'inférence sur une plateforme Edge.

### Chaîne de traitement

```text
Dataset Thick_Ghana
        │
        ▼
Validation et nettoyage
        │
        ▼
Séparation Train / Validation / Test
        │
        ▼
Entraînement des modèles YOLO
        │
        ▼
Évaluation comparative
        │
        ▼
Sélection de YOLOv26n
        │
        ▼
Exportation et optimisation
        │
        ▼
TFLite INT8
        │
        ▼
Raspberry Pi 3 B+
        │
        ▼
Prétraitement
        │
        ▼
Inférence
        │
        ▼
Post-traitement
        │
        ▼
Détection des parasites
