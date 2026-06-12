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

- **Date:** TBD (Notebook 03)
- **Architecture:**
  - Input Layer: 9 features (after encoding)
  - Hidden Layer 1: 64 neurons, ReLU
  - Hidden Layer 2: 32 neurons, ReLU
  - Output Layer: 1 neuron, Sigmoid
- **Training Config:**
  - Optimizer: Adam (lr=0.001)
  - Loss: Binary Cross Entropy
  - Epochs: 50
  - Batch Size: 64
- **Imbalance Handling:** None
- **Regularization:** None
- **Expected Weakness:** Will likely predict mostly legitimate (class imbalance), recall will be low
- **Metrics:** To be filled after training

| Metric | Train | Validation | Test |
|---|---|---|---|
| Loss | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1 | — | — | — |
| ROC-AUC | — | — | — |
| PR-AUC | — | — | — |

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

*Last updated: 2026-06-12 | Phase: 1 – Planning*
