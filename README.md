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

paludisme-edge-ai/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── .gitignore
├── requirements-train.txt
├── requirements-edge.txt
│
├── configs/
│   ├── data.yaml.example
│   ├── training.yaml
│   ├── export.yaml
│   └── benchmark.yaml
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── kaggle/
│   ├── 01_validate_dataset.py
│   ├── 02_prepare_dataset.py
│   ├── 03_train_yolov8n.py
│   ├── 04_train_yolo11n.py
│   ├── 05_train_yolov26n.py
│   ├── 06_evaluate_models.py
│   ├── 07_confusion_matrix.py
│   ├── 08_performance_curves.py
│   ├── 09_export_onnx.py
│   ├── 10_export_tflite_float16.py
│   ├── 11_export_tflite_dynamic_range.py
│   └── 12_export_tflite_int8.py
│
├── raspberry_pi/
│   ├── 01_system_info.py
│   ├── 02_check_model.py
│   ├── 03_inference.py
│   ├── 04_benchmark.py
│   ├── 05_benchmark_resolutions.py
│   ├── 06_compare_threads.py
│   ├── 07_memory_monitor.py
│   ├── 08_image_preprocessing.py
│   ├── 09_postprocessing.py
│   ├── 10_energy_measurement.py
│   └── 11_run_all_benchmarks.sh
│
├── web/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── src/
│   ├── preprocessing.py
│   ├── inference.py
│   ├── postprocessing.py
│   └── utils.py
│
├── results/
│   ├── README.md
│   └── .gitkeep
│
└── docs/
    ├── reproduction.md
    ├── architecture.md
    ├── deployment_raspberry_pi.md
    ├── benchmark.md
    └── troubleshooting.md
