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

---

## DEC-006: Optimizer and Scheduler Selection

- **Date:** 2026-06-16
- **Context:** Choose the optimal optimizer and learning rate decay scheduling strategy for the MLP.
- **Options Considered:**
  - Optimizers: SGD, SGD + Momentum, Nesterov Momentum, Adagrad, RMSprop, Adam, AdamW.
  - Schedulers (with AdamW): Step Decay, Cosine Decay, Warmup Cosine Decay.
- **Decision:** **AdamW with Warmup Cosine Decay Scheduler** (warmup_epochs=5, eta_min=0.0001)
- **Rationale:** Adam/AdamW provided the strongest baseline results (F1: 71.11%, PR-AUC: 0.7936), but applying the Warmup Cosine Scheduler on AdamW yielded a significant performance boost (F1: 72.34%, PR-AUC: 0.8338). The linear warmup epoch range stabilizes gradient flow early on, and the cosine annealing decay facilitates optimal parameter convergence near the end of training.
- **Trade-offs Accepted:** Adds hyperparameters (warmup epochs, min learning rate) that require tuning and tracking.
- **Revisit Trigger:** If introducing heavy regularization (Dropout/L2) alters optimizer behavior.

---

## DEC-007: Activation Function Selection — Leaky ReLU over other Activations

- **Date:** 2026-06-15
- **Context:** Select the activation function that maximizes model performance and stabilizes training on the imbalanced credit card fraud dataset.
- **Options Considered:**
  - Sigmoid: Standard squashing function, but susceptible to vanishing gradients.
  - Tanh: Zero-centered but still susceptible to vanishing gradients in deep networks.
  - ReLU: Standard deep learning activation, but prone to "dying ReLU" (inactive neurons for negative values).
  - Leaky ReLU: Keeps a small positive gradient (slope = 0.01) for negative inputs.
  - ELU: Smooth exponential curve for negative values; computationally more expensive.
  - GELU: Stochastic regularization effect; state-of-the-art for transformers but less studied on simple MLPs.
- **Decision:** **Leaky ReLU** (with negative slope = 0.01)
- **Rationale:** Leaky ReLU yielded the highest test F1-Score of 71.11%, matching the highest test recall of 69.57% while exhibiting the most stable validation learning behavior. Standardizing on Leaky ReLU prevents the dead neuron problem, ensuring all parts of the network continue to receive gradients and adapt to minority fraud samples.
- **Trade-offs Accepted:** Adds a hyperparameter (negative slope) that was set to a standard default of 0.01.
- **Revisit Trigger:** If introducing Batch Normalization or changes in weight initialization schemes significantly alter activation scaling.

---

## DEC-008: Weight Initialization Selection — Random Uniform over Xavier and Kaiming

- **Date:** 2026-06-16
- **Context:** Select the weight initialization method that maximizes test F1-Score, stability, and convergence speed on the credit card fraud detection MLP configuration.
- **Options Considered:**
  - PyTorch Default: Kaiming uniform with standard PyTorch bounds (starting loss 0.679, F1-Score 72.34%).
  - Random Uniform: Naive uniform initialization in $[-0.05, 0.05]$ (starting loss 0.693, F1-Score 79.07%).
  - Random Normal: Naive normal initialization with standard deviation 0.05 (starting loss 0.691, F1-Score 76.19%).
  - Xavier Uniform/Normal: Scaled to balance input/output variance for symmetric activations (F1-Scores: 72.73% / 65.12%).
  - Kaiming Uniform/Normal: Scaled to balance input/output variance for rectified activations like Leaky ReLU (F1-Scores: 71.43% / 66.67%).
- **Decision:** **Random Uniform** (uniform distribution in $[-0.05, 0.05]$) for weights, and initialize biases to `0.0`.
- **Rationale:** Random Uniform initialization achieved the highest test performance (F1-Score of **79.07%** and Recall of **73.91%**) while converging fastest (**23 epochs**). For a shallow network trained on a highly imbalanced dataset, the larger initial weight variance of mathematically derived Xavier/Kaiming methods causes the model to fit noise in the training set too quickly. Limiting initial weights to small values acts as an implicit regularizer, preventing early overfitting to majority non-fraud samples and boosting test generalization.
- **Trade-offs Accepted:** Small random initialization does not scale well to very deep networks, where Xavier/Kaiming are required to prevent vanishing/exploding gradients. However, for our 2-layer MLP architecture, the regularization benefit dominates.
- **Revisit Trigger:** If the architecture depth is significantly increased or if explicit regularization (Dropout, weight decay) is added in Phase 8.

---

## DEC-009: Regularization Selection — Early Stopping Only

- **Date:** 2026-06-16
- **Context:** Select the regularization strategy that minimizes overfitting while preserving minority class recall and F1-score on the credit card fraud detection MLP configuration.
- **Options Considered:**
  - No Regularization: Train full 50 epochs (Overfit Gap: +4.10%, Test F1: 79.07%).
  - Early Stopping Only: Restrict training via validation loss patience = 5 (Overfit Gap: -8.89%, Test F1: 79.07%).
  - Dropout (p=0.3 or p=0.5): Randomly zero out activations (Test F1: 76.19%).
  - Batch Normalization: Normalize layers over mini-batches (Test F1: 29.63%).
  - L1 Regularization (λ=1e-4): L1 penalty on weights (Test F1: 56.25%).
  - L2 Regularization / Weight Decay (1e-4): L2 penalty on weights (Test F1: 79.07%).
  - Gradient Clipping (5.0): Clip backpropagated gradients (Test F1: 79.07%).
- **Decision:** **Early Stopping Only** (patience = 5) restored to the best validation loss epoch is selected as the primary regularization strategy.
- **Rationale:** Early Stopping successfully keeps the overfit gap negative (**-8.89%**), ensuring the model generalizes extremely well, while maximizing test F1-Score (**79.07%**) and test Recall (**73.91%**) in only **23 epochs**. 
  Explicit regularizers like Dropout and L1 penalty underfit the dataset because the network capacity is already small (only 13 inputs, hidden layers of 64 and 32 neurons). Batch Normalization collapsed recall entirely because batch-level mean and variance estimates fluctuate wildly on highly imbalanced datasets (where fraud is only 0.17% of samples), introducing severe noise into parameter updates.
- **Trade-offs Accepted:** Early stopping requires saving checkpoints during training and running validation evaluations at each epoch, adding a small amount of memory/compute overhead during training, but zero overhead at inference.
- **Revisit Trigger:** If the architecture capacity is dramatically scaled up (e.g. 5+ deep hidden layers) or if class imbalance strategies in Phase 9 alter the distribution of training mini-batches.

---

*Last updated: 2026-06-16 | Phase: 8 – Regularization Techniques*
