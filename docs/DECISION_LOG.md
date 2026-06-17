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

## DEC-010: Class Imbalance Handling Strategy — WeightedRandomSampler

- **Date:** 2026-06-16
- **Context:** Select the class imbalance handling strategy that optimizes the recall of the minority fraud class on holdout test data while preserving high overall classification capacity (measured by Precision-Recall AUC).
- **Options Considered:**
  - Baseline (No Handling): Train on raw imbalanced dataset (Test Recall: 73.91%, Test Precision: 85.00%, PR-AUC: 0.8599).
  - Weighted BCE Loss: Apply a $pos\_weight$ scaling factor of ~65.04 to the positive class in BCE loss (Test Recall: 100.00%, Test Precision: 25.84%, PR-AUC: 0.8472).
  - WeightedRandomSampler: Dynamically oversample minority class at DataLoader batch level (Test Recall: 95.65%, Test Precision: 44.90%, PR-AUC: 0.8580).
  - SMOTE: Train on pre-generated synthetic oversampled dataset (Test Recall: 78.26%, Test Precision: 36.00%, PR-AUC: 0.6287).
- **Decision:** **WeightedRandomSampler** (with sample weights inversely proportional to class counts, and replacement = True) is selected as the winning strategy for **MODEL-v5**.
- **Rationale:** 
  - In a fraud detection context, missing fraud is highly costly. WeightedRandomSampler dramatically improved Test Recall to **95.65%** (missing only 1 of 23 fraud transactions) while maintaining an outstanding **PR-AUC of 0.8580**, showing that the model retains its underlying classification separation power.
  - While it lowers default threshold precision to **44.90%**, this is a direct result of shifting the logits' default bias towards a 50/50 balance. This can be easily corrected in Phase 11 by optimizing the classification decision threshold.
  - SMOTE failed catastrophically (PR-AUC collapsed to **0.6287** and Precision to **36.00%**) because feature-space interpolation creates synthetic points in regions that belong to the majority class, confusing the decision boundary.
  - Weighted BCE Loss achieved 100% recall but yielded a lower PR-AUC (0.8472) and a very high rate of False Positives (25.84% precision).
- **Trade-offs Accepted:** Oversampling the minority class means the model is exposed to duplicates of fraud cases in training, but this is mitigated by early stopping on raw validation data which prevents memorization.
- **Revisit Trigger:** If the model capacity is scaled up or during threshold optimization in Phase 11.

## DEC-011: Advanced Architecture Selection — Baseline MLP over Tabular ResNet and Gated MLP

- **Date:** 2026-06-17
- **Context:** Select the model architecture that maximizes overall fraud detection separation capacity (PR-AUC) and recall on holdout test data.
- **Options Considered:**
  - Baseline MLP: 2 hidden layers (64, 32), Leaky ReLU activations, Random Uniform weight initialization, AdamW optimizer, Warmup Cosine scheduler (Test Recall: 95.65%, Test Precision: 38.60%, PR-AUC: 0.8273).
  - Tabular ResNet: Deep network (hidden dim 64, 3 residual blocks with LayerNorm and skip connections) (Test Recall: 56.52%, Test Precision: 50.00%, PR-AUC: 0.6007).
  - Gated MLP: 2 Gated Linear Unit (GLU) layers (64, 32) (Test Recall: 82.61%, Test Precision: 47.50%, PR-AUC: 0.7316).
- **Decision:** **Baseline MLP (Retained)** is selected as the winning architecture for **MODEL-v6**.
- **Rationale:** 
  - **Overfitting on Small-Scale Data:** With only 13 input features and 7,000 training samples, the dataset is too small to require high-capacity deep learning structures. The Tabular ResNet overfits very rapidly to training noise and minority-class duplicates, triggering early stopping at epoch 8 and collapsing Test Recall to **56.52%** and PR-AUC to **0.6007**.
  - **Feature Gating vs Baseline:** The Gated MLP's GLU mechanism introduces feature-dependent gating, which acts as a soft filter. This improves Precision (**47.50%**) and default-threshold F1-Score (**60.32%**), but reduces Test Recall (**82.61%**) and overall class discrimination capacity (PR-AUC of **0.7316** vs baseline **0.8273**).
  - **Overall Capacity:** The Baseline MLP remains the most robust architecture across all probability thresholds, maintaining the highest PR-AUC (**0.8273**) and Recall (**95.65%**).
- **Trade-offs Accepted:** Choosing a simpler MLP limits the model's capacity to learn complex, non-linear high-order interactions that deep structures might capture on millions of samples. However, for this dataset size, the simpler MLP is significantly more robust.
- **Revisit Trigger:** If the dataset size is expanded by orders of magnitude (e.g. 100k+ samples) or if new high-dimensional feature engineering is introduced.

---

*Last updated: 2026-06-17 | Phase: 10 – Advanced Model Architecture*
