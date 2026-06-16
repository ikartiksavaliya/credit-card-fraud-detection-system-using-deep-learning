# Deep Credit Card Fraud Detection System
## PROJECT MASTER DOCUMENT

> **Version:** 1.0.0 | **Started:** 2026-06-12 | **Framework:** PyTorch

---

## 1. PROJECT OVERVIEW

### Business Problem
Credit card fraud costs the global financial industry **$32 billion+ annually**. Every fraudulent transaction not caught results in:
- Direct monetary loss (chargeback + goods/services lost)
- Operational overhead for dispute resolution (~$40–$100 per case)
- Reputational damage and customer churn
- Regulatory penalties for systemic failures

The goal of this system is to **detect fraudulent transactions with maximum recall** while controlling false positives to an acceptable level — because in fraud detection, **missing a fraud (False Negative) is almost always more costly than a false alarm (False Positive)**.

### Learning Goal
Master the full deep learning stack from first principles — neurons, backpropagation, optimizers, regularization — by building a real-world system.

### Portfolio Goal
Produce a project that demonstrates:
- Professional software engineering (modular code, notebooks, documentation)
- Deep understanding of DL theory (not just API calls)
- Rigorous experimentation and scientific reasoning
- Business-aware modeling decisions

---

## 2. DATASET

| Property | Value |
|---|---|
| **File** | `data/credit_card_fraud_10k.csv` |
| **Rows** | 10,001 (10,000 transactions + header) |
| **Features** | 9 input + 1 target |
| **Task** | Binary Classification |

### Feature Glossary

| Feature | Type | Description | Business Meaning |
|---|---|---|---|
| `transaction_id` | Integer | Unique row identifier | Drop before training |
| `amount` | Float | Transaction amount in USD | High-value transactions → higher fraud risk |
| `transaction_hour` | Integer [0–23] | Hour of day transaction occurred | Late-night transactions are higher risk |
| `merchant_category` | Categorical | Type of merchant (Electronics, Travel, Food, Grocery, Clothing) | Electronics and Travel → higher fraud rate |
| `foreign_transaction` | Binary [0/1] | Whether it occurred in a foreign country | Foreign = higher risk |
| `location_mismatch` | Binary [0/1] | Billing address ≠ transaction location | Strong fraud signal |
| `device_trust_score` | Integer [0–100] | Trust score of the device used | Low score = untrusted device = higher risk |
| `velocity_last_24h` | Integer | Number of transactions in last 24h | High velocity = unusual activity pattern |
| `cardholder_age` | Integer | Age of cardholder | May correlate with fraud vulnerability |
| `is_fraud` | Binary [0/1] | **TARGET**: Whether transaction is fraudulent | 0=Legitimate, 1=Fraud |

---

## 3. ARCHITECTURE DECISIONS

### Hybrid Architecture
- **Notebooks**: Learning + experimentation + visualization + narrative
- **Python modules** (`src/`): Reusable, testable, importable utilities

### Framework
- **PyTorch** for all deep learning (manual, explicit, educational)
- **scikit-learn** for preprocessing and classical baselines
- **imbalanced-learn** for SMOTE

### Design Principles
1. Explain before implement
2. Experiment before conclude
3. Measure before optimize
4. Document everything

---

## 4. PHASE ROADMAP

| Phase | Notebook | Topic | Status |
|---|---|---|---|
| 1 | — | Planning, Dataset Inspection, Documentation | ✅ Completed |
| 2 | 01_eda.ipynb | Exploratory Data Analysis | ✅ Completed |
| 3 | 02_preprocessing.ipynb | Data Preprocessing + Feature Engineering | ✅ Completed |
| 4 | 03_baseline_mlp.ipynb | Neural Network Foundations + Baseline MLP | ✅ Completed |
| 5 | 04_activation_study.ipynb | Activation Function Deep Dive | ✅ Completed |
| 6 | 05_optimizer_study.ipynb | Optimizer Comparison Study | ✅ Completed |
| 7 | 06_initialization_study.ipynb | Weight Initialization Study | ✅ Completed |
| 8 | 07_regularization_study.ipynb | Regularization Techniques | ✅ Completed |
| 9 | 08_class_imbalance_study.ipynb | Class Imbalance Strategies | ⬜ Pending |
| 10 | 09_advanced_model.ipynb | Advanced Architecture | ⬜ Pending |
| 11 | 10_threshold_optimization.ipynb | Business-Aware Threshold Tuning | ⬜ Pending |
| 12 | 11_final_report.ipynb | Final Report + Portfolio Assembly | ⬜ Pending |

---

## 5. SUCCESS CRITERIA

| Metric | Target | Business Rationale |
|---|---|---|
| **Recall (Fraud)** | ≥ 0.85 | Missing fraud is very costly |
| **Precision (Fraud)** | ≥ 0.70 | Some false alarms acceptable |
| **PR-AUC** | ≥ 0.80 | Key metric for imbalanced data |
| **ROC-AUC** | ≥ 0.90 | Overall discriminative power |
| **F1 Score (Fraud)** | ≥ 0.75 | Balance of precision/recall |

---

## 6. KEY CONTACTS & REFERENCES

- Dataset: `data/credit_card_fraud_10k.csv` (local, synthetic)
- Framework: PyTorch >= 2.0
- Python: >= 3.10
