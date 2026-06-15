# INTERVIEW NOTES

> Curated deep learning interview Q&A organized by topic.
> Every question here was chosen because it is commonly asked in ML Engineer / Data Scientist interviews.
> Written in your own words — not copy-pasted definitions.

---

## 🧠 NEURAL NETWORK FOUNDATIONS

### Q1: What is a neuron in a neural network? How does it differ from a biological neuron?

An artificial neuron is a mathematical function that models the basic signaling of a biological neuron. It takes a vector of inputs $x$, computes a weighted sum plus a bias ($z = w^T x + b$), and passes the result through a non-linear activation function ($y = \text{activation}(z)$).

**Biological Analogy:**
- **Dendrites** correspond to input features $x$.
- **Synapses** correspond to the weights $w$, scaling the input signals.
- **Soma (cell body)** corresponds to the summation and bias addition.
- **Axon** corresponds to the output signal $y$, which fires when the combined input exceeds a threshold (modelled by the activation function).

**Key Differences:**
- **Complexity**: Biological neurons are complex, stochastic, non-linear electrochemical systems with complex internal states. Artificial neurons are simple, deterministic, continuous mathematical functions.
- **Timing**: Biological neurons communicate using discrete electrical spikes (spiking neural networks) and process information asynchronously in continuous time. Artificial neurons communicate using real-valued activation levels and process information in synchronous discrete steps.

---

### Q2: Why do we need non-linear activation functions? What happens if we don't use them?
*(To be filled after Phase 5)*

**Key points to cover:**
- Without activation: network collapses to a single linear transformation regardless of depth
- Proof: composition of linear functions is linear → depth buys nothing
- Non-linearity = ability to approximate any continuous function (Universal Approximation Theorem)

---

### Q3: Explain backpropagation. What is the chain rule and why does it matter?

Backpropagation is the fundamental algorithm used to calculate the gradient of the loss function with respect to the weights and biases of a neural network. It propagates the error signal backward from the output layer to the input layer using the **Chain Rule** of calculus.

**The Chain Rule:**
If a variable $y$ depends on $u$, which in turn depends on $x$, then the rate of change of $y$ with respect to $x$ is:
$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

In a neural network, for a weight $W$ at a given layer:
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial W}$$
Where $L$ is the loss, $a$ is the activation output, and $z$ is the weighted linear sum.

**Why it matters:**
Backpropagation allows us to calculate the exact analytical gradients for all parameters in a single backward pass. Without it, we would have to estimate gradients numerically (e.g., using finite differences), which requires executing the forward pass twice for every single parameter. For modern networks with millions or billions of parameters, this is computationally impossible. Backpropagation makes deep learning computationally feasible.

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

We use Binary Cross Entropy (BCE) instead of Mean Squared Error (MSE) for binary classification due to optimization efficiency and probabilistic alignment:

1. **Gradient Saturation (Vanishing Gradients)**: Binary classifiers typically use a Sigmoid output layer to predict probabilities. Sigmoid saturates (its derivative $\sigma'(z) \to 0$) for very large positive or negative logits. If we use MSE loss, the gradient is scaled by $\sigma'(z)$. If the model makes a highly confident but incorrect prediction, the gradient becomes near-zero, meaning the model ceases to learn. BCE loss mathematically cancels out the $\sigma'(z)$ term in the gradient:
   $$\frac{\partial L_{BCE}}{\partial z} = \sigma(z) - y$$
   The gradient is proportional to the simple prediction error ($\hat{y} - y$). If the error is large, the gradient is large, driving rapid learning.
2. **Probabilistic Consistency**: BCE represents the negative log-likelihood of a Bernoulli distribution. Minimizing BCE is equivalent to Maximum Likelihood Estimation (MLE) for binary classification. MSE assumes a continuous Gaussian target, which does not match binary $\{0, 1\}$ classes.

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

In credit card fraud detection:
- **Precision** is the percentage of flagged transactions that are actually fraudulent:
  $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
- **Recall** is the percentage of actual fraud cases that our model successfully identifies:
  $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

**The Trade-off:**
- **High Recall Focus**: Lowering the classification threshold catches more fraud (fewer False Negatives), but flags many legitimate transactions as fraudulent (more False Positives). This lowers Precision, causing customer annoyance (declined cards) and increasing support center volume.
- **High Precision Focus**: Raising the threshold ensures that we only flag transactions that are highly suspicious (fewer False Positives). This keeps customer friction low but misses actual fraud cases (more False Negatives), resulting in high direct chargeback losses.

The business optimal threshold must be optimized to balance this trade-off based on the financial and operational costs of False Positives vs False Negatives.

### Q25: What is threshold optimization? How would you choose the optimal classification threshold?
*(To be filled after Phase 11)*

### Q26: How would you explain your model's decision to a business stakeholder who doesn't understand ML?
*(To be filled after Phase 12)*

---

*Last updated: 2026-06-13 | Phase: 4 – Baseline MLP*
