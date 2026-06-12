# TOPIC COVERAGE MATRIX

> This matrix shows which notebooks and source files cover each topic.
> It ensures no topic is studied in isolation — each must appear in code, experiments, and docs.

---

## Coverage Legend

| Symbol | Meaning |
|---|---|
| 📖 | Explained (narrative / theory) |
| 💻 | Implemented (code) |
| 📊 | Evaluated (experiment results) |
| 📝 | Documented (docs entry) |

---

## Matrix

| Topic | 01_eda | 02_prep | 03_baseline | 04_act | 05_opt | 06_init | 07_reg | 08_imb | 09_adv | 10_thresh | src/ | docs/ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Neurons / Layers** | | | 📖💻 | | | | | | | | models.py | CHECKLIST |
| **Weights & Biases** | | | 📖💻 | | | 💻 | | | | | models.py | CHECKLIST |
| **Forward Pass** | | | 📖💻📊 | | | | | | | | models.py | CHECKLIST |
| **Sigmoid** | | | | 📖💻📊 | | | | | | | models.py | CHECKLIST |
| **Tanh** | | | | 📖💻📊 | | | | | | | models.py | CHECKLIST |
| **ReLU** | | | 💻 | 📖💻📊 | | | | | | | models.py | CHECKLIST |
| **Leaky ReLU** | | | | 📖💻📊 | | | | | | | models.py | CHECKLIST |
| **ELU** | | | | 📖💻📊 | | | | | | | models.py | CHECKLIST |
| **GELU** | | | | 📖💻📊 | | | | | | | models.py | CHECKLIST |
| **Binary Cross Entropy** | | | 📖💻 | 💻 | 💻 | 💻 | 💻 | | | | training.py | CHECKLIST |
| **Weighted BCE** | | | | | | | | 📖💻📊 | | | training.py | CHECKLIST |
| **Backprop / Chain Rule** | | | 📖 | | | | | | | | — | CHECKLIST |
| **Computational Graph** | | | 📖💻 | | | | | | | | — | CHECKLIST |
| **Gradient Flow** | | | 📖💻 | | | | | | 💻 | | training.py | CHECKLIST |
| **Batch GD / Mini-Batch** | | | 📖💻 | | | | | | | | training.py | CHECKLIST |
| **SGD** | | | | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **Learning Rate** | | | 📖 | | 📖💻 | | | | | | training.py | CHECKLIST |
| **Step Decay** | | | | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **Cosine Decay** | | | | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **Warmup** | | | | | 📖💻📊 | | | | 💻 | | training.py | CHECKLIST |
| **Momentum** | | | | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **Nesterov** | | | | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **Adagrad** | | | | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **RMSProp** | | | | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **Adam** | | | 💻 | | 📖💻📊 | | | | | | training.py | CHECKLIST |
| **AdamW** | | | | | 📖💻📊 | | | | 💻 | | training.py | CHECKLIST |
| **Xavier Init** | | | | | | 📖💻📊 | | | | | models.py | CHECKLIST |
| **He Init** | | | | | | 📖💻📊 | | | | | models.py | CHECKLIST |
| **Batch Normalization** | | | | | | | 📖💻📊 | | 💻 | | models.py | CHECKLIST |
| **Gradient Clipping** | | | | | | | 📖💻 | | 💻 | | training.py | CHECKLIST |
| **Dropout** | | | | | | | 📖💻📊 | | 💻 | | models.py | CHECKLIST |
| **L1 / L2 Regularization** | | | | | | | 📖💻📊 | | | | training.py | CHECKLIST |
| **Weight Decay** | | | | | | | 📖💻📊 | | | | training.py | CHECKLIST |
| **Early Stopping** | | | | | | | 📖💻📊 | | 💻 | | training.py | CHECKLIST |
| **Overfitting / Underfitting** | | | 📖 | | | | 📊 | | | | — | CHECKLIST |
| **Vanishing Gradients** | | | 📖 | 💻📊 | | 📊 | | | | | — | CHECKLIST |
| **Exploding Gradients** | | | 📖 | | | | 💻 | | | | — | CHECKLIST |
| **Baseline (no imbalance)** | | | 📊 | | | | | 📖💻📊 | | | — | CHECKLIST |
| **WeightedRandomSampler** | | | | | | | | 📖💻📊 | | | training.py | CHECKLIST |
| **SMOTE** | | 📖💻 | | | | | | 📖💻📊 | | | preprocessing.py | CHECKLIST |
| **Confusion Matrix** | | | 💻📊 | | | | | | | | evaluation.py | CHECKLIST |
| **Precision/Recall/F1** | | | 💻📊 | | | | | | | | evaluation.py | CHECKLIST |
| **ROC-AUC** | | | 💻📊 | | | | | | | | evaluation.py | CHECKLIST |
| **PR-AUC** | | | 💻📊 | | | | | | | | evaluation.py | CHECKLIST |
| **Threshold Optimization** | | | | | | | | | | 📖💻📊 | evaluation.py | CHECKLIST |

---

*Last updated: 2026-06-13 | Phase: 3 – Data Preprocessing + Feature Engineering*
