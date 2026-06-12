# MODEL EVOLUTION LOG

> This document tracks every version of the model — what changed, why, and what improved.
> Think of it as a git history for the model architecture itself.

---

## Evolution Timeline

```
Baseline MLP
    ↓ (Add BatchNorm)
Stabilized MLP
    ↓ (Tune activation: ReLU → GELU)
Optimized MLP
    ↓ (Add Dropout + Weight Decay)
Regularized MLP
    ↓ (Handle class imbalance: WeightedBCE / SMOTE)
Balanced MLP
    ↓ (Add residual connections, deeper architecture)
Advanced MLP
    ↓ (Threshold optimization for business objective)
Production Model
```

---

## MODEL-v0: Baseline MLP

- **Date:** 2026-06-13 (Notebook 03)
- **Architecture:**
  - Input Layer: 13 features (after encoding)
  - Hidden Layer 1: 64 neurons, ReLU
  - Hidden Layer 2: 32 neurons, ReLU
  - Output Layer: 1 neuron, Sigmoid (logits used in training)
- **Training Config:**
  - Optimizer: Adam (lr=0.001)
  - Loss: Binary Cross Entropy (via BCEWithLogitsLoss)
  - Epochs: 50 (Early stopped at epoch 25, best weights from epoch 20)
  - Batch Size: 64
- **Imbalance Handling:** None
- **Regularization:** None
- **Expected Weakness:** Susceptible to class imbalance; performance represents a baseline before imbalance handling strategies.
- **Metrics:** Filled after training

| Metric | Train | Validation | Test |
|---|---|---|---|
| Loss | 0.0080 | 0.0136 | 0.0212 |
| Precision | 0.9787 | 0.9091 | 0.6957 |
| Recall | 0.8762 | 0.8696 | 0.6957 |
| F1 | 0.9246 | 0.8889 | 0.6957 |
| ROC-AUC | 0.9997 | 0.9973 | 0.9951 |
| PR-AUC | 0.9853 | 0.9047 | 0.7779 |

---

## MODEL-v1: Activation Study Winner

- **Date:** TBD (Notebook 04)
- **Change:** Replace ReLU with the best-performing activation from study
- **Hypothesis:** Non-standard activations (GELU/ELU) may perform better on this tabular fraud data
- **Architecture:** Same as v0 but with best activation
- **Metrics:** To be filled

---

## MODEL-v2: Optimizer Study Winner

- **Date:** TBD (Notebook 05)
- **Change:** Replace Adam with the best-performing optimizer
- **Hypothesis:** AdamW may outperform vanilla Adam due to proper weight decay
- **Metrics:** To be filled

---

## MODEL-v3: Initialization Study Winner

- **Date:** TBD (Notebook 06)
- **Change:** Apply best weight initialization scheme
- **Hypothesis:** He initialization works best with ReLU-family activations
- **Metrics:** To be filled

---

## MODEL-v4: Regularized MLP

- **Date:** TBD (Notebook 07)
- **Change:** Add dropout + BatchNorm + weight decay based on regularization study
- **Expected Impact:** Reduced overfitting, better generalization
- **Metrics:** To be filled

---

## MODEL-v5: Class-Balanced MLP

- **Date:** TBD (Notebook 08)
- **Change:** Apply best class imbalance strategy
- **Expected Impact:** Higher recall on fraud class
- **Metrics:** To be filled

---

## MODEL-v6: Advanced Architecture

- **Date:** TBD (Notebook 09)
- **Change:** Deeper network with skip connections, attention-like mechanisms, or similar
- **Rationale:** Explore whether added capacity helps on this tabular dataset
- **Metrics:** To be filled

---

## MODEL-Final: Business-Optimized Model

- **Date:** TBD (Notebook 10–11)
- **Change:** Optimal threshold tuned for business objective (minimize total loss)
- **Metrics:** To be filled — this is the final production model

---

*Last updated: 2026-06-13 | Phase: 4 – Baseline MLP*
