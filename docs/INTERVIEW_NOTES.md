# INTERVIEW NOTES

> Curated deep learning interview Q&A organized by topic.
> Every question here was chosen because it is commonly asked in ML Engineer / Data Scientist interviews.
> Written in your own words — not copy-pasted definitions.

---

## 🧠 NEURAL NETWORK FOUNDATIONS

### Q1: What is a neuron in a neural network? How does it differ from a biological neuron?
*(To be filled after Phase 4)*

**Key points to cover:**
- Mathematical formulation: y = activation(w·x + b)
- Biological analogy (dendrites = inputs, axon = output, synapse = weights)
- Key difference: biological neurons are complex stochastic systems; artificial neurons are simple deterministic functions

---

### Q2: Why do we need non-linear activation functions? What happens if we don't use them?
*(To be filled after Phase 5)*

**Key points to cover:**
- Without activation: network collapses to a single linear transformation regardless of depth
- Proof: composition of linear functions is linear → depth buys nothing
- Non-linearity = ability to approximate any continuous function (Universal Approximation Theorem)

---

### Q3: Explain backpropagation. What is the chain rule and why does it matter?
*(To be filled after Phase 4)*

---

## ⚡ ACTIVATION FUNCTIONS

### Q4: What is the vanishing gradient problem with Sigmoid/Tanh?
*(To be filled after Phase 5)*

### Q5: Why is ReLU preferred over Sigmoid in hidden layers?
*(To be filled after Phase 5)*

### Q6: What is the dying ReLU problem? How do Leaky ReLU and ELU solve it?
*(To be filled after Phase 5)*

### Q7: What is GELU? Why is it used in modern transformers?
*(To be filled after Phase 5)*

---

## 📉 LOSS FUNCTIONS

### Q8: Why do we use Binary Cross Entropy for binary classification, not MSE?
*(To be filled after Phase 4)*

### Q9: How does weighted BCE help with class imbalance?
*(To be filled after Phase 9)*

---

## 🚀 OPTIMIZERS

### Q10: Explain Adam optimizer. What are β1 and β2?
*(To be filled after Phase 6)*

### Q11: What is the difference between Adam and AdamW?
*(To be filled after Phase 6)*

### Q12: Why does SGD with momentum sometimes generalize better than Adam?
*(To be filled after Phase 6)*

### Q13: What is gradient descent? Explain batch vs mini-batch vs stochastic.
*(To be filled after Phase 6)*

---

## 🎲 WEIGHT INITIALIZATION

### Q14: Why is weight initialization important? What happens if all weights are initialized to zero?
*(To be filled after Phase 7)*

### Q15: Explain Xavier and He initialization. When do you use each?
*(To be filled after Phase 7)*

---

## 🛡️ TRAINING STABILITY

### Q16: What does Batch Normalization do? Why does it help training?
*(To be filled after Phase 8)*

### Q17: What is gradient clipping and when is it necessary?
*(To be filled after Phase 8)*

---

## 🔒 REGULARIZATION

### Q18: What is the difference between L1 and L2 regularization? Which produces sparse weights?
*(To be filled after Phase 8)*

### Q19: How does Dropout regularize a neural network? What is the intuition?
*(To be filled after Phase 8)*

### Q20: What is the difference between weight decay and L2 regularization in the context of Adam optimizer?
*(To be filled after Phase 8)*

---

## ⚖️ CLASS IMBALANCE

### Q21: What is class imbalance and why is accuracy a misleading metric for fraud detection?
*(To be filled after Phase 9)*

### Q22: Compare SMOTE, class weights, and WeightedRandomSampler. When would you use each?
*(To be filled after Phase 9)*

### Q23: What is PR-AUC and why is it preferred over ROC-AUC for imbalanced datasets?
*(To be filled after Phase 9)*

---

## 📊 EVALUATION & BUSINESS

### Q24: Explain precision vs recall trade-off in a fraud detection context.
*(To be filled after Phase 4)*

### Q25: What is threshold optimization? How would you choose the optimal classification threshold?
*(To be filled after Phase 11)*

### Q26: How would you explain your model's decision to a business stakeholder who doesn't understand ML?
*(To be filled after Phase 12)*

---

*Last updated: 2026-06-12 | Phase: 1 – Planning*
