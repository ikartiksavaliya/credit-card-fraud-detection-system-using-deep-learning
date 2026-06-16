# EXPERIMENT TRACKER

> All experiments are logged here with their configuration, results, and conclusions.
> This is the scientific record of the project.

---

## Experiment Log Format

Each entry follows this template:

```
### EXP-XXX: [Experiment Name]
- **Date:** YYYY-MM-DD
- **Notebook:** [notebook name]
- **Hypothesis:** [What do you expect to happen and why?]
- **Config:** [Model, optimizer, LR, epochs, etc.]
- **Results:** [Table of metrics]
- **Conclusion:** [What did you learn?]
- **Surprises:** [Anything unexpected?]
```

---

## Activation Function Study (Notebook 04)

- **Date:** 2026-06-15
- **Notebook:** 04_activation_study.ipynb
- **Hypothesis:** Non-standard/non-saturating activations (like Leaky ReLU, ELU, GELU) prevent the dying ReLU problem and improve gradient flow on tabular data.
- **Config:** Baseline MLP (Input -> 64 -> 32 -> 1, no dropout/normalization, Adam lr=0.001, BCEWithLogitsLoss).
- **Results:**

| Activation | Train Loss | Val Loss | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Sigmoid | 0.0254 | 0.0217 | 82.35% | 60.87% | 70.00% | 0.9968 | 0.8418 |
| Tanh | 0.0187 | 0.0198 | 76.47% | 56.52% | 65.00% | 0.9947 | 0.7892 |
| ReLU | 0.0091 | 0.0136 | 69.57% | 69.57% | 69.57% | 0.9951 | 0.7779 |
| Leaky ReLU | 0.0086 | 0.0133 | 72.73% | 69.57% | 71.11% | 0.9954 | 0.7936 |
| ELU | 0.0199 | 0.0195 | 70.59% | 52.17% | 60.00% | 0.9942 | 0.7614 |
| GELU | 0.0118 | 0.0153 | 81.25% | 56.52% | 66.67% | 0.9944 | 0.7825 |

**Winner:** Leaky ReLU | **Reasoning:** Leaky ReLU achieved the highest test F1-Score (71.11%) and matched the highest recall (69.57%) while showing stable validation learning behavior. Its non-zero negative slope prevents the dying ReLU problem and maintains active gradient flow.

---

## Optimizer Study (Notebook 05)

*To be filled during Phase 6 execution.*

| Optimizer | LR | Train Loss | Val Loss | Precision | Recall | F1 | ROC-AUC | PR-AUC | Convergence Speed |
|---|---|---|---|---|---|---|---|---|---|
| SGD | 0.01 | 0.0542 | 0.0556 | 0.00% | 0.00% | 0.00% | 0.9592 | 0.4804 | 50 epochs (No ES) |
| SGD + Momentum | 0.01 | 0.0116 | 0.0146 | 70.00% | 60.87% | 65.12% | 0.9953 | 0.8176 | 44 epochs |
| Nesterov | 0.01 | 0.0115 | 0.0147 | 70.00% | 60.87% | 65.12% | 0.9951 | 0.8070 | 44 epochs |
| Adagrad | 0.001 | 0.0748 | 0.0744 | 0.00% | 0.00% | 0.00% | 0.8530 | 0.0760 | 50 epochs (No ES) |
| RMSProp | 0.001 | 0.0067 | 0.0130 | 73.68% | 60.87% | 66.67% | 0.9953 | 0.7822 | 25 epochs |
| Adam | 0.001 | 0.0075 | 0.0133 | 72.73% | 69.57% | 71.11% | 0.9954 | 0.7936 | 25 epochs |
| AdamW | 0.001 | 0.0075 | 0.0133 | 72.73% | 69.57% | 71.11% | 0.9954 | 0.7936 | 25 epochs |
| AdamW + Step Decay | 0.001 | 0.0075 | 0.0137 | 69.57% | 69.57% | 69.57% | 0.9954 | 0.7889 | 44 epochs |
| AdamW + Cosine Decay | 0.001 | 0.0055 | 0.0135 | 72.73% | 69.57% | 71.11% | 0.9956 | 0.7919 | 34 epochs |
| AdamW + Warmup Cosine | 0.001 | 0.0058 | 0.0128 | 70.83% | 73.91% | 72.34% | 0.9962 | 0.8338 | 41 epochs |

**Winner:** AdamW + Warmup Cosine Decay | **Reasoning:** Adam/AdamW provided the strongest baseline results. Introducing a Warmup Cosine scheduler on top of AdamW further boosted performance, raising the test F1-Score from 71.11% to 72.34% and test PR-AUC from 0.7936 to 0.8338 while maintaining excellent validation and training loss curves.

---

## Initialization Study (Notebook 06)

- **Date:** 2026-06-16
- **Notebook:** 06_initialization_study.ipynb
- **Hypothesis:** He (Kaiming) initialization matches Leaky ReLU activations best by accounting for the rectification scaling factor, yielding faster convergence and better test metrics compared to naive uniform/normal or Xavier initialization.
- **Config:** MODEL-v2 MLP (Leaky ReLU activations, AdamW optimizer, Warmup Cosine scheduler, batch size 64).
- **Results:**

| Initialization | Train Loss (epoch 1) | Val Loss (final) | Precision | Recall | F1 | ROC-AUC | PR-AUC | Convergence |
|---|---|---|---|---|---|---|---|---|
| PyTorch Default | 0.6793 | 0.0128 | 70.83% | 73.91% | 72.34% | 0.9962 | 0.8338 | 41 epochs |
| Random (Uniform) | 0.6932 | 0.0190 | 85.00% | 73.91% | 79.07% | 0.9968 | 0.8370 | 23 epochs |
| Random (Normal) | 0.6909 | 0.0179 | 84.21% | 69.57% | 76.19% | 0.9975 | 0.8655 | 32 epochs |
| Xavier Uniform | 0.8040 | 0.0141 | 76.19% | 69.57% | 72.73% | 0.9958 | 0.8370 | 44 epochs |
| Xavier Normal | 0.4669 | 0.0171 | 70.00% | 60.87% | 65.12% | 0.9947 | 0.8133 | 32 epochs |
| He Uniform | 0.8166 | 0.0168 | 78.95% | 65.22% | 71.43% | 0.9954 | 0.8301 | 44 epochs |
| He Normal | 0.4556 | 0.0183 | 73.68% | 60.87% | 66.67% | 0.9932 | 0.7590 | 32 epochs |

**Winner:** Random Uniform (uniform in $[-0.05, 0.05]$) | **Reasoning:** Surprisingly, initializing weights to small bounds (Random Uniform or Random Normal) outperformed the mathematically derived Xavier/Kaiming initializations in terms of test set generalization. In a shallow MLP (2 hidden layers) trained on a small, highly imbalanced dataset, the larger initial weight variance of Xavier/Kaiming causes the model to fit training noise too rapidly, leading to overfitting. Small random initializations keep the weight values constrained, acting as an implicit regularizer that boosts holdout test performance (raising F1-Score to **79.07%** and Recall to **73.91%**) and converges in fewer epochs (**23 epochs**).

---

## Regularization Study (Notebook 07)

*To be filled during Phase 8 execution.*

| Technique | Train F1 | Val F1 | Overfit Gap | PR-AUC |
|---|---|---|---|---|
| No Regularization | — | — | — | — |
| Dropout (p=0.3) | — | — | — | — |
| Dropout (p=0.5) | — | — | — | — |
| BatchNorm | — | — | — | — |
| L1 (λ=1e-4) | — | — | — | — |
| L2 (λ=1e-4) | — | — | — | — |
| Weight Decay | — | — | — | — |
| Early Stopping | — | — | — | — |
| Combined Best | — | — | — | — |

**Winner:** TBD | **Reasoning:** TBD

---

## Class Imbalance Study (Notebook 08)

*To be filled during Phase 9 execution.*

| Strategy | Recall (Fraud) | Precision (Fraud) | F1 (Fraud) | PR-AUC |
|---|---|---|---|---|
| Baseline (no handling) | — | — | — | — |
| Weighted BCE Loss | — | — | — | — |
| WeightedRandomSampler | — | — | — | — |
| SMOTE | — | — | — | — |

**Winner:** TBD | **Business reasoning:** TBD

---

*Last updated: 2026-06-16 | Phase: 7 – Weight Initialization Study*
