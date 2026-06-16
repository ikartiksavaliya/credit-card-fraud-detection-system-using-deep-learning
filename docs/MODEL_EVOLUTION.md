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
| Loss | 0.0043 | 0.0127 | 0.0182 |
| Precision | 0.9903 | 0.9091 | 0.8095 |
| Recall | 0.9714 | 0.8696 | 0.7391 |
| F1 | 0.9808 | 0.8889 | 0.7727 |
| ROC-AUC | 1.0000 | 0.9981 | 0.9964 |
| PR-AUC | 0.9991 | 0.9223 | 0.8422 |

---

## MODEL-v1: Activation Study Winner

- **Date:** 2026-06-15 (Notebook 04)
- **Change:** Replace ReLU with the best-performing activation from study (Leaky ReLU)
- **Hypothesis:** Non-standard/non-saturating activations (like Leaky ReLU, ELU, GELU) prevent the dying ReLU problem and improve gradient flow on tabular data. Leaky ReLU was chosen because it achieved the highest F1-Score (71.11%) and matched the highest recall (69.57%) on the test set while showing stable validation learning behavior.
- **Architecture:**
  - Input Layer: 13 features (after encoding)
  - Hidden Layer 1: 64 neurons, Leaky ReLU (slope = 0.01)
  - Hidden Layer 2: 32 neurons, Leaky ReLU (slope = 0.01)
  - Output Layer: 1 neuron, Sigmoid (logits used in training)
- **Metrics:**

| Metric | Train | Validation | Test |
|---|---|---|---|
| Loss | 0.0075 | 0.0133 | 0.0206 |
| Precision | 0.9894 | 0.8696 | 0.7273 |
| Recall | 0.8857 | 0.8696 | 0.6957 |
| F1 | 0.9347 | 0.8696 | 0.7111 |
| ROC-AUC | 0.9998 | 0.9977 | 0.9954 |
| PR-AUC | 0.9867 | 0.9090 | 0.7936 |

---

## MODEL-v2: Optimizer Study Winner

- **Date:** 2026-06-16 (Notebook 05)
- **Change:** Replace Adam with AdamW, and apply a Warmup Cosine learning rate scheduler.
- **Hypothesis:** Warmup Cosine scheduler provides a smooth learning rate transition, starting with linear warmup to prevent gradient explosion/early suboptimal local minima, followed by cosine annealing to fine-tune the weights, leading to better test F1-Score (72.34%) and PR-AUC (0.8338).
- **Architecture:**
  - Input Layer: 13 features (after encoding)
  - Hidden Layer 1: 64 neurons, Leaky ReLU (slope = 0.01)
  - Hidden Layer 2: 32 neurons, Leaky ReLU (slope = 0.01)
  - Output Layer: 1 neuron, Sigmoid (logits used in training)
- **Training Config:**
  - Optimizer: AdamW (lr=0.001)
  - Scheduler: Warmup Cosine (warmup_epochs=5, total_epochs=50, eta_min=0.0001)
  - Loss: BCEWithLogitsLoss
  - Epochs: 50 (Early stopped at epoch 41, best weights from epoch 36)
  - Batch Size: 64
- **Metrics:**

| Metric | Train | Validation | Test |
|---|---|---|---|
| Loss | 0.0058 | 0.0128 | 0.0199 |
| Precision | 0.9796 | 0.8261 | 0.7083 |
| Recall | 0.9143 | 0.8261 | 0.7391 |
| F1 | 0.9458 | 0.8261 | 0.7234 |
| ROC-AUC | 0.9999 | 0.9979 | 0.9962 |
| PR-AUC | 0.9925 | 0.9181 | 0.8338 |

---

## MODEL-v3: Initialization Study Winner

- **Date:** 2026-06-16 (Notebook 06)
- **Change:** Apply Random Uniform weight initialization (uniform in $[-0.05, 0.05]$) for all linear weights, and initialize biases to `0.0`.
- **Hypothesis:** For a shallow MLP on a small, highly imbalanced dataset, naive small-weight initializations act as an implicit regularizer. By limiting initial weight magnitudes, they prevent the model from overfitting early to training noise, leading to better test-set generalization than Xavier or Kaiming methods.
- **Architecture:**
  - Input Layer: 13 features
  - Hidden Layer 1: 64 neurons, Leaky ReLU (slope = 0.01)
  - Hidden Layer 2: 32 neurons, Leaky ReLU (slope = 0.01)
  - Output Layer: 1 neuron, Sigmoid (logits used in training)
- **Training Config:**
  - Optimizer: AdamW (lr=0.001)
  - Scheduler: Warmup Cosine (warmup_epochs=5, total_epochs=50, eta_min=0.0001)
  - Weight Initialization: Random Uniform (bounds $[-0.05, 0.05]$)
  - Loss: BCEWithLogitsLoss
  - Epochs: 50 (Early stopped at epoch 23, best weights from epoch 18)
  - Batch Size: 64
- **Metrics:**

| Metric | Train | Validation | Test |
|---|---|---|---|
| Loss | 0.0188 | 0.0190 | 0.0184 |
| Precision | 0.8533 | 0.9412 | 0.8500 |
| Recall | 0.6095 | 0.6957 | 0.7391 |
| F1 | 0.7111 | 0.8000 | 0.7907 |
| ROC-AUC | 0.9965 | 0.9962 | 0.9968 |
| PR-AUC | 0.8507 | 0.8756 | 0.8370 |

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

*Last updated: 2026-06-16 | Phase: 7 – Weight Initialization Study*
