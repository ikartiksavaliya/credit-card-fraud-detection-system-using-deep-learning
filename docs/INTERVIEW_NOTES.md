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

If we do not use non-linear activation functions, a neural network collapses mathematically into a single linear transformation, regardless of how many layers it has. 

* **Mathematical Proof:**
  Let a 2-layer network have inputs $x$, weight matrices $W_1, W_2$, and biases $b_1, b_2$. Without activation functions, the output $y$ is:
  $$y = W_2(W_1 x + b_1) + b_2$$
  $$y = (W_2 W_1) x + (W_2 b_1 + b_2)$$
  If we define a single equivalent weight matrix $W' = W_2 W_1$ and a single bias vector $b' = W_2 b_1 + b_2$, the equation becomes:
  $$y = W' x + b'$$
  This is a simple linear function. No matter how deep the network is, the composition of linear functions remains linear. Thus, a multi-layer linear network has no more representational capacity than a single-layer linear model.
* **Universal Approximation:**
  Non-linear activation functions allow the network to learn complex, non-linear decision boundaries. According to the **Universal Approximation Theorem**, a feedforward network with a single hidden layer and a non-linear activation function can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision. Non-linearities are what give neural networks their representation capacity and power.

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

* **Sigmoid and Tanh Derivatives:**
  The derivative of the Sigmoid function $\sigma(z)$ is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, which reaches a maximum value of only $0.25$ (at $z=0$). The derivative of Tanh $g(z)$ is $g'(z) = 1 - \tanh^2(z)$, which has a maximum value of $1.0$ (at $z=0$).
* **How Gradients Vanish:**
  During backpropagation, the gradient of the loss with respect to a weight in an early layer is calculated using the chain rule, which multiplies the derivatives of the activation functions of all subsequent layers:
  $$\frac{\partial L}{\partial W^{[1]}} = \text{error} \times \prod_{i=2}^{L} a'_{i}(z^{[i]}) \times \dots$$
  Since the derivatives of Sigmoid and Tanh are bounded and typically much less than $1.0$ (especially when inputs $z$ are large positive or negative and the function saturates, i.e., $a' \approx 0$), multiplying many small numbers together across multiple layers causes the gradient to shrink exponentially as it flows backward.
* **Consequence:**
  The weights in the earliest layers of the network receive near-zero updates, meaning they learn extremely slowly or stop learning altogether. This limits the practical depth of networks using Sigmoid/Tanh activations.

### Q5: Why is ReLU preferred over Sigmoid in hidden layers?

ReLU ($\text{ReLU}(z) = \max(0, z)$) is preferred over Sigmoid in hidden layers for three primary reasons:
1. **Constant Gradient (No Saturation on Positive Domain):** For any positive input ($z > 0$), the derivative of ReLU is exactly $1.0$. This prevents the gradient from shrinking during backpropagation, effectively solving the vanishing gradient problem for positive inputs and enabling the training of much deeper networks.
2. **Computational Efficiency:** Sigmoid requires expensive exponential operations ($e^{-z}$) and division, whereas ReLU can be computed extremely fast using a simple thresholding operation at $0$ (which is highly optimized on CPUs/GPUs).
3. **Representational Sparsity:** ReLU outputs exactly $0.0$ for all negative inputs ($z \le 0$). This results in sparse activations (meaning only a subset of neurons fire for any given input), which reduces computational overhead and helps prevent overfitting.

### Q6: What is the dying ReLU problem? How do Leaky ReLU and ELU solve it?

* **The Dying ReLU Problem:**
  Since ReLU outputs exactly $0.0$ and has a derivative of exactly $0.0$ for any negative input ($z < 0$), if a neuron gets updated such that it receives negative inputs for the entire training dataset, it will output $0.0$ and its gradient will be exactly $0.0$ for all training samples. Consequently, its weights will never update again during backpropagation. The neuron effectively "dies" and becomes permanently inactive, reducing the network's capacity.
* **Leaky ReLU Solution:**
  Leaky ReLU introduces a small, non-zero slope ($\alpha$, typically $0.01$) for negative inputs:
  $$\text{Leaky ReLU}(z) = \max(\alpha z, z)$$
  This ensures that even when $z < 0$, the gradient is a small non-zero value ($\alpha$), allowing the neuron to continue receiving weight updates and eventually recover.
* **ELU Solution:**
  ELU (Exponential Linear Unit) uses an exponential curve for negative inputs:
  $$\text{ELU}(z) = z \text{ if } z > 0 \text{ else } \alpha(e^z - 1)$$
  ELU provides a small non-zero gradient for negative inputs to prevent dead neurons, while smoothing the transition around zero. It also pushes the mean activation close to zero, which speeds up convergence.

### Q7: What is GELU? Why is it used in modern transformers?

* **Definition of GELU:**
  GELU (Gaussian Error Linear Unit) scales the input $z$ by its cumulative distribution function (CDF) under a standard normal distribution:
  $$\text{GELU}(z) = z \cdot \Phi(z) = z \cdot P(X \le z) \text{ where } X \sim \mathcal{N}(0, 1)$$
* **Why GELU is Used:**
  Unlike ReLU, which strictly truncates negative inputs at $0$, or Leaky ReLU, which scales them by a fixed constant, GELU weights the input deterministically but probabilistically based on its value.
  - If the input is very negative, $\Phi(z) \to 0$, making the output near $0$.
  - If the input is very positive, $\Phi(z) \to 1$, making the output near $z$.
  - Near zero, GELU provides a smooth, non-monotonic curve that has a small curvature. Because it is smooth and has non-zero derivatives everywhere, it allows gradient flow even for negative inputs.
  - **In Transformers:** Transformers (e.g., BERT, GPT, ViT) rely heavily on layer normalization and feedforward sub-networks. GELU's smooth gradient flow helps stabilize self-attention optimization, prevents dead zones, and empirically leads to faster training and better generalization performance than ReLU on natural language and vision processing tasks.

---

## 📉 LOSS FUNCTIONS

### Q8: Why do we use Binary Cross Entropy for binary classification, not MSE?

We use Binary Cross Entropy (BCE) instead of Mean Squared Error (MSE) for binary classification due to optimization efficiency and probabilistic alignment:

1. **Gradient Saturation (Vanishing Gradients)**: Binary classifiers typically use a Sigmoid output layer to predict probabilities. Sigmoid saturates (its derivative $\sigma'(z) \to 0$) for very large positive or negative logits. If we use MSE loss, the gradient is scaled by $\sigma'(z)$. If the model makes a highly confident but incorrect prediction, the gradient becomes near-zero, meaning the model ceases to learn. BCE loss mathematically cancels out the $\sigma'(z)$ term in the gradient:
   $$\frac{\partial L_{BCE}}{\partial z} = \sigma(z) - y$$
   The gradient is proportional to the simple prediction error ($\hat{y} - y$). If the error is large, the gradient is large, driving rapid learning.
2. **Probabilistic Consistency**: BCE represents the negative log-likelihood of a Bernoulli distribution. Minimizing BCE is equivalent to Maximum Likelihood Estimation (MLE) for binary classification. MSE assumes a continuous Gaussian target, which does not match binary $\{0, 1\}$ classes.

### Q9: How does weighted BCE help with class imbalance?

Weighted Binary Cross Entropy (BCE) addresses class imbalance by scaling the loss penalty associated with the minority class.

In standard BCE, the loss for a single sample is:
$$L = - [y \log \hat{y} + (1 - y) \log (1 - \hat{y})]$$
Where $y \in \{0, 1\}$ is the true label and $\hat{y} = \sigma(z)$ is the predicted probability. When class imbalance is severe, the total loss and its gradients are dominated by the majority class $0$.

Weighted BCE introduces a positive-class scaling factor $w_{\text{pos}}$ (commonly configured as $w_{\text{pos}} = \text{negative\_samples} / \text{positive\_samples}$):
$$L_{\text{weighted}} = - [w_{\text{pos}} \cdot y \log \hat{y} + (1 - y) \log (1 - \hat{y})]$$

**Mathematical Mechanism:**
1. **Gradient Amplification:** The gradient of the loss with respect to the input logit $z$ is:
   $$\frac{\partial L_{\text{weighted}}}{\partial z} = \begin{cases} \sigma(z) & \text{if } y = 0 \\ w_{\text{pos}}(\sigma(z) - 1) & \text{if } y = 1 \end{cases}$$
   If the model makes an error on a minority sample ($y=1$ but predicts a low probability $\sigma(z) \to 0$), the backpropagated gradient is multiplied by $w_{\text{pos}}$. This forces a massive parameter update to correct the error. In our dataset, this update is $\approx 65\times$ larger than an update for a majority class error.
2. **Logit Shift:** Penalizing False Negatives more heavily shifts the model's logits higher, effectively moving the decision boundary closer to the majority class and increasing Recall at the cost of Precision.

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

* **Class Imbalance:** Occurs when one class (the majority class, e.g., legitimate transactions) vastly outnumbers the other class (the minority class, e.g., fraud) in the training dataset. In our credit card dataset, fraud represents only ~1.5% of the data (and ~0.17% in real-world systems).
* **Why Accuracy is Misleading:** Accuracy measures the percentage of correct predictions:
  $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$$
  If a dataset has 98.5% legitimate transactions and 1.5% fraud, a naive classifier that predicts "legitimate" for every transaction will achieve **98.5% accuracy**. However, its fraud detection rate (Recall) is **0%**. Accuracy rewards the model for predicting the majority class correctly, completely hiding the fact that it fails to detect the critical minority class.

### Q22: Compare SMOTE, class weights, and WeightedRandomSampler. When would you use each?

| Technique | Level | Description | Pros | Cons | Best Used For |
|---|---|---|---|---|---|
| **Class Weights / Weighted BCE** | Loss | Scales loss gradients for minority class errors during backpropagation. | Simple; no memory overhead; doesn't duplicate data. | Can cause unstable gradients if weights are extremely large; some batches may contain zero fraud samples. | Medium imbalance; constraint-heavy environments. |
| **WeightedRandomSampler** | Batch | Samples training data with replacement to create balanced (e.g. 50/50) mini-batches. | Dynamically balances batches; guarantees stable, frequent minority gradients; keeps original data intact. | Duplicates minority samples, which can cause overfitting without proper regularization. | Deep Learning models on highly imbalanced tabular or image data. |
| **SMOTE** | Data | Synthesizes new minority samples by interpolating between nearest neighbors in feature space. | Expands the minority decision boundary instead of repeating existing points. | Can create noisy/invalid samples in sparse, high-dimensional tabular space, collapsing Precision. | Shallow ML models (e.g. SVM, Random Forests) on moderate tabular imbalance. |

### Q23: What is PR-AUC and why is it preferred over ROC-AUC for imbalanced datasets?

* **Definitions:**
  * **ROC Curve** plots True Positive Rate (Recall) vs False Positive Rate (FPR):
    $$\text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}$$
  * **PR Curve** plots Precision vs Recall (TPR).
* **Why PR-AUC is Preferred:**
  * FPR includes True Negatives ($\text{TN}$) in its denominator. On highly imbalanced datasets, $\text{TN}$ is extremely large. Even if the model makes many False Positives (e.g., flagging 500 legitimate transactions as fraud), the FPR remains very small because the denominator is huge. Thus, the ROC curve appears highly optimistic, showing a near-perfect AUC (e.g., 0.99) even when the model makes a large number of false alarms.
  * Precision is normalized by predicted positives ($\text{TP} + \text{FP}$). Because it does not include $\text{TN}$, Precision directly exposes the frequency of false alarms relative to true fraud cases. A model with 500 FP and only 20 TP will have a very low Precision ($\approx 3.8\%$), causing the PR curve and PR-AUC to drop dramatically.
  * Therefore, PR-AUC provides a much more honest and sensitive metric for evaluation when the minority class is rare and of primary interest.

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

*Last updated: 2026-06-16 | Phase: 9 – Class Imbalance Strategies*
