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

*To be filled during Phase 5 execution.*

| Activation | Train Loss | Val Loss | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Sigmoid | — | — | — | — | — | — | — |
| Tanh | — | — | — | — | — | — | — |
| ReLU | — | — | — | — | — | — | — |
| Leaky ReLU | — | — | — | — | — | — | — |
| ELU | — | — | — | — | — | — | — |
| GELU | — | — | — | — | — | — | — |

**Winner:** TBD | **Reasoning:** TBD

---

## Optimizer Study (Notebook 05)

*To be filled during Phase 6 execution.*

| Optimizer | LR | Train Loss | Val Loss | Precision | Recall | F1 | ROC-AUC | PR-AUC | Convergence Speed |
|---|---|---|---|---|---|---|---|---|---|
| SGD | 0.01 | — | — | — | — | — | — | — | — |
| SGD + Momentum | 0.01 | — | — | — | — | — | — | — | — |
| Nesterov | 0.01 | — | — | — | — | — | — | — | — |
| Adagrad | 0.001 | — | — | — | — | — | — | — | — |
| RMSProp | 0.001 | — | — | — | — | — | — | — | — |
| Adam | 0.001 | — | — | — | — | — | — | — | — |
| AdamW | 0.001 | — | — | — | — | — | — | — | — |

**Winner:** TBD | **Reasoning:** TBD

---

## Initialization Study (Notebook 06)

*To be filled during Phase 7 execution.*

| Initialization | Train Loss (epoch 1) | Val Loss (final) | Recall | F1 | Convergence |
|---|---|---|---|---|---|
| Random (Uniform) | — | — | — | — | — |
| Random (Normal) | — | — | — | — | — |
| Xavier | — | — | — | — | — |
| He | — | — | — | — | — |

**Winner:** TBD | **Reasoning:** TBD

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

*Last updated: 2026-06-12 | Phase: 1 – Planning*
