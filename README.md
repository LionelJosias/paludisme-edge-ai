# 🦟 Paludisme Edge AI — Détection embarquée des parasites du paludisme

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLOv26n-Ultralytics-orange.svg)](https://docs.ultralytics.com/)
[![Edge](https://img.shields.io/badge/Edge-Raspberry%20Pi%203%20B%2B-red.svg)]
[![Runtime](https://img.shields.io/badge/Runtime-LiteRT%20%2F%20TFLite-green.svg)]

Projet académique consacré à la **détection automatisée des parasites du paludisme dans des images de frottis sanguins épais**, en combinant l'intelligence artificielle, les modèles YOLO légers et l'Edge Computing.

L'objectif est d'étudier la faisabilité d'une solution capable d'effectuer localement la détection des parasites sur une plateforme embarquée à ressources limitées, sans dépendre d'un serveur distant ou du Cloud pour l'inférence.

> **Statut : preuve expérimentale de faisabilité.**  
> Ce projet est réalisé dans un cadre académique et ne constitue pas un dispositif médical ni un outil de diagnostic clinique validé.

---

# 1. Présentation du projet

Le projet couvre l'ensemble de la chaîne de traitement, depuis la préparation des données jusqu'à l'inférence sur une plateforme Edge.

```mermaid
flowchart LR
    A[Lacuna Malaria Dataset<br/>Thick_Ghana]
    --> B[Préparation et nettoyage]
    --> C[Train / Validation / Test]
    --> D[Entraînement des modèles YOLO]
    --> E[Évaluation comparative]
    --> F[Sélection de YOLOv26n]
    --> G[Exportation]
    --> H[Optimisation]
    --> I[Déploiement Raspberry Pi 3 B+]
    --> J[Prétraitement]
    --> K[Inférence]
    --> L[Post-traitement]
    --> M[Détection des parasites] diagnostic médical.
