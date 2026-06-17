# 🔐 Deep Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

> A portfolio-grade deep learning project for credit card fraud detection — built systematically from first principles using PyTorch, with rigorous experimentation, professional documentation, and business-aware decision-making.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Learning Roadmap](#learning-roadmap)
- [Key Results](#key-results)
- [Setup & Installation](#setup--installation)
- [How to Use](#how-to-use)
- [Documentation](#documentation)

---

## Project Overview

Credit card fraud costs the global financial industry **$32 billion+ annually**. This project builds a production-quality fraud detection system while systematically teaching deep learning from first principles.

### Dual Objective
1. **Production Quality:** Build a real fraud detection system with rigorous experimentation
2. **Educational Depth:** Understand every concept — neurons, backprop, optimizers, regularization — not just use APIs

### Key Features
- 📚 **11 learning notebooks** covering EDA through advanced modeling
- 🔬 **4 dedicated study notebooks**: Activation, Optimizer, Initialization, Regularization
- ⚖️ **4 class imbalance strategies** compared head-to-head
- 📊 **Complete experiment tracking** with metrics logged for every study
- 💼 **Business-aware modeling** with threshold optimization and cost-benefit analysis
- 🗂️ **Professional documentation** system (8 specialized docs files)

---

## Dataset

| Property | Value |
|---|---|
| Source | Synthetic (local CSV) |
| Size | 10,000 transactions |
| Features | 9 input features |
| Target | Binary (0=Legitimate, 1=Fraud) |
| Imbalance | ~3–10% fraud rate (analyzed in EDA) |

**Features:** `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`

---

## Project Structure

```
credit-card-fraud-detection-system/
│
├── data/
│   └── credit_card_fraud_10k.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Feature Engineering & Preprocessing
│   ├── 03_baseline_mlp.ipynb          # NN Foundations + Baseline Model
│   ├── 04_activation_study.ipynb      # Sigmoid vs Tanh vs ReLU vs GELU
│   ├── 05_optimizer_study.ipynb       # SGD vs Adam vs AdamW (7 optimizers)
│   ├── 06_initialization_study.ipynb  # Random vs Xavier vs He
│   ├── 07_regularization_study.ipynb  # Dropout vs BatchNorm vs L1/L2
│   ├── 08_class_imbalance_study.ipynb # Baseline vs WeightedBCE vs SMOTE
│   ├── 09_advanced_model.ipynb        # Advanced Architecture
│   ├── 10_threshold_optimization.ipynb # Business-Optimal Threshold
│   └── 11_final_report.ipynb          # Final Report + Portfolio Assembly
│
├── src/
│   ├── preprocessing.py    # Data cleaning, encoding, scaling, SMOTE
│   ├── models.py           # MLP architectures (configurable)
│   ├── training.py         # Training loop, schedulers, early stopping
│   ├── evaluation.py       # Metrics, ROC, PR curves, confusion matrix
│   └── utils.py            # Seeding, plotting, logging utilities
│
├── docs/
│   ├── PROJECT_MASTER.md       # Master reference document
│   ├── LEARNING_CHECKLIST.md   # Topic completion tracking
│   ├── TOPIC_COVERAGE_MATRIX.md # Which notebook covers which topic
│   ├── EXPERIMENT_TRACKER.md   # All experiment results
│   ├── DECISION_LOG.md         # Technical decisions + rationale
│   ├── MODEL_EVOLUTION.md      # Model version history
│   ├── INTERVIEW_NOTES.md      # Deep learning interview Q&A
│   └── BUSINESS_IMPACT.md      # Business context + cost analysis
│
├── outputs/
│   ├── figures/            # All saved plots
│   ├── models/             # Saved model checkpoints (.pt files)
│   └── reports/            # Generated HTML/PDF reports
│
└── README.md
```

---

## Learning Roadmap

| Phase | Topic | Notebook | Status |
|---|---|---|---|
| 1 | Planning, Dataset Inspection, Documentation | — | ✅ Complete |
| 2 | Exploratory Data Analysis | 01_eda | ✅ Complete |
| 3 | Preprocessing + Feature Engineering | 02_preprocessing | ✅ Complete |
| 4 | NN Foundations + Baseline MLP | 03_baseline_mlp | ✅ Complete |
| 5 | Activation Function Study | 04_activation_study | ✅ Complete |
| 6 | Optimizer Study | 05_optimizer_study | ✅ Complete |
| 7 | Initialization Study | 06_initialization_study | ✅ Complete |
| 8 | Regularization Study | 07_regularization_study | ✅ Complete |
| 9 | Class Imbalance Study | 08_class_imbalance_study | ✅ Complete |
| 10 | Advanced Architecture | 09_advanced_model | ✅ Complete |
| 11 | Threshold Optimization | 10_threshold_optimization | ✅ Complete |
| 12 | Final Report | 11_final_report | ✅ Complete |

---

## Key Results

Evaluation results of model progression on the holdout test set (under default threshold $T = 0.500$):

| Model Version | Recall | Precision | F1-Score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **Baseline MLP** (Phase 4) | 73.9% | 81.0% | 77.3% | 0.8422 | 0.9964 |
| **+ Best Activation** (Leaky ReLU) | 73.9% | 70.8% | 72.3% | 0.8338 | 0.9962 |
| **+ Best Optimizer** (AdamW + Warmup Cosine) | 69.6% | 72.7% | 71.1% | 0.7936 | 0.9954 |
| **+ Best Init** (Random Normal) | 69.6% | 84.2% | 76.2% | 0.8655 | 0.9975 |
| **+ Regularization** (Early Stopping / Grad Clip) | 73.9% | 85.0% | 79.1% | 0.8370 | 0.9968 |
| **+ Class Imbalance** (WeightedRandomSampler) | 95.7% | 44.9% | 61.1% | 0.8580 | 0.9967 |
| **Final (Threshold-Opt)** (MLP w/ Sampler, T=0.500) | 95.7% | 38.6% | 55.0% | 0.8273 | 0.9960 |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Requirements
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Documentation

All documentation lives in `docs/`. Start with:
- [**`FINAL_REPORT.md`**](docs/FINAL_REPORT.md) — Summary portfolio presentation of final results and system design
- [**`FINAL_AUDIT_REPORT.md`**](docs/FINAL_AUDIT_REPORT.md) — Comprehensive engineering, scientific, and portfolio audit
- [`PROJECT_MASTER.md`](docs/PROJECT_MASTER.md) — Project overview and decisions
- [`LEARNING_CHECKLIST.md`](docs/LEARNING_CHECKLIST.md) — Topic completion tracker
- [`EXPERIMENT_TRACKER.md`](docs/EXPERIMENT_TRACKER.md) — All experiment results
- [`BUSINESS_IMPACT.md`](docs/BUSINESS_IMPACT.md) — Business context and cost analysis
- [`INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) — Deep learning Q&A for interviews
- [`MODEL_EVOLUTION.md`](docs/MODEL_EVOLUTION.md) — Detailed model evolution history

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for learning deep learning from first principles.*
