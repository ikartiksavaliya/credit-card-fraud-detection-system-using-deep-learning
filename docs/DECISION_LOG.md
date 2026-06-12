# DECISION LOG

> Every major technical decision is logged here with its rationale.
> This is critical for understanding WHY choices were made — not just WHAT was chosen.

---

## Decision Format

```
### DEC-XXX: [Decision Name]
- **Date:** YYYY-MM-DD
- **Context:** [What problem triggered this decision?]
- **Options Considered:** [What alternatives existed?]
- **Decision:** [What was chosen?]
- **Rationale:** [Why this option?]
- **Trade-offs Accepted:** [What did you give up?]
- **Revisit Trigger:** [Under what conditions should this be reconsidered?]
```

---

## DEC-001: Framework Selection — PyTorch over TensorFlow/Keras

- **Date:** 2026-06-12
- **Context:** Need a deep learning framework for the project.
- **Options Considered:**
  - TensorFlow/Keras: High-level, easy to prototype, production-ready
  - PyTorch: Explicit, Pythonic, research-standard, better for learning internals
  - JAX: Cutting-edge but steep learning curve
- **Decision:** PyTorch
- **Rationale:** PyTorch's explicit tensor operations, dynamic computation graphs, and `nn.Module` API expose the internals of deep learning clearly. For a learning-focused project, seeing `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` manually is far more educational than Keras's `model.fit()`. Industry trend also strongly favors PyTorch in research and increasingly in production.
- **Trade-offs Accepted:** More verbose code; no Keras-style rapid prototyping.
- **Revisit Trigger:** If the project requires production deployment to mobile/edge (then TFLite may be needed).

---

## DEC-002: Notebook-First Architecture

- **Date:** 2026-06-12
- **Context:** How should the project be organized — pure scripts, pure notebooks, or hybrid?
- **Options Considered:**
  - Pure scripts: Reproducible but not educational or interactive
  - Pure notebooks: Interactive but messy for reusable code
  - Hybrid: Notebooks for exploration + `src/` for reusable modules
- **Decision:** Hybrid architecture (notebooks + src/)
- **Rationale:** Notebooks provide narrative, visualization, and experimentation. Python modules in `src/` provide reusable, importable, maintainable code. This mirrors professional ML engineering workflows.
- **Trade-offs Accepted:** Slightly more overhead maintaining two code locations.
- **Revisit Trigger:** Never — this is the correct architecture for a portfolio project.

---

## DEC-003: Primary Evaluation Metric — PR-AUC over ROC-AUC

- **Date:** 2026-06-12
- **Context:** The dataset has significant class imbalance (fraud is rare). Which metric should guide model selection?
- **Options Considered:**
  - Accuracy: Misleading on imbalanced data (a model predicting all-legitimate gets ~97% accuracy)
  - ROC-AUC: Robust but optimistic on imbalanced datasets because it includes TN in denominator
  - PR-AUC: Focuses only on the positive (fraud) class; more honest for imbalanced data
  - F1 Score: Good single-number summary but threshold-dependent
- **Decision:** **PR-AUC as primary metric**, ROC-AUC as secondary, F1 and Recall for reporting
- **Rationale:** When fraud rate is ~3–10%, the large number of true negatives inflates ROC-AUC. PR-AUC only evaluates performance on the fraud class, making it the most honest measure of model usefulness.
- **Trade-offs Accepted:** PR-AUC is harder to interpret for non-technical stakeholders.
- **Revisit Trigger:** If dataset becomes more balanced (>20% fraud rate).

---

## DEC-004: Dataset Split Strategy

- **Date:** 2026-06-12
- **Context:** How to split 10,000 samples for train/val/test?
- **Options Considered:**
  - 80/20 train/test: Standard but no validation set
  - 70/15/15 train/val/test: Common three-way split
  - 80/10/10 train/val/test: More training data, less evaluation data
  - Time-based split: Simulates production (past → present)
- **Decision:** **70/15/15 stratified split**
- **Rationale:** With 10,000 samples, 7,000 training examples is adequate. Stratification ensures the fraud rate is preserved in each split. The test set is held out until final evaluation only.
- **Trade-offs Accepted:** Less training data than 80/20, but more reliable validation signal.
- **Revisit Trigger:** If a timestamp column is available, switch to temporal split.

---

## DEC-005: Class Imbalance — Investigate Before Acting

- **Date:** 2026-06-12
- **Context:** The dataset has class imbalance. Should we oversample, undersample, or use loss weighting?
- **Decision:** Dedicate Notebook 08 to a full comparative study of 4 strategies before choosing.
- **Rationale:** Blindly applying SMOTE or weighting without understanding why is bad engineering. The study will reveal the true imbalance ratio and which strategy best fits this dataset.
- **Trade-offs Accepted:** Slower progression, but much deeper understanding.
- **Revisit Trigger:** After seeing imbalance ratio in EDA.

---

*Last updated: 2026-06-12 | Phase: 1 – Planning*
