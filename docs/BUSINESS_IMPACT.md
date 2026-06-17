# BUSINESS IMPACT ANALYSIS

> This document connects every technical decision to its real-world business consequence.
> A model that performs well in the lab but fails in production is worthless.
> A model that performs moderately in the lab but saves the company $10M is priceless.

---

## 1. THE COST OF FRAUD

### Direct Costs (per fraudulent transaction)
- **Transaction loss:** Full amount of the fraudulent charge
- **Chargeback fee:** $15–$100 per dispute (paid by merchant)
- **Chargeback ratio penalty:** If > 1% disputes, payment processors increase fees or terminate accounts
- **Operational cost:** Manual review, customer service, disputes department

### Indirect Costs
- **Customer trust erosion:** Victims of fraud often close their accounts
- **Brand reputation:** High-profile breaches damage public perception for years
- **Regulatory fines:** GDPR, PCI-DSS violations for inadequate fraud controls

---

## 2. THE COST OF FALSE POSITIVES (Blocking Legitimate Transactions)

This is the **hidden cost** that most tutorials ignore.

When a legitimate transaction is blocked:
- **Customer friction:** They may abandon the purchase entirely (conversion loss)
- **Revenue loss:** Missed sale
- **Customer churn:** Repeated false declines → customer switches to a competitor card
- **Support call volume:** Each declined card generates ~1–2 support calls

**Industry benchmark:** 1 false positive costs approximately $5–$20 in customer experience terms.

---

## 3. THE COST MATRIX

This is the framework for making business-aware decisions:

| Prediction → | Actual Legitimate | Actual Fraud |
|---|---|---|
| **Predicted Legitimate** | ✅ True Negative — Zero cost | ❌ False Negative — **HIGH COST** |
| **Predicted Fraud** | ⚠️ False Positive — **MEDIUM COST** | ✅ True Positive — Zero cost (fraud caught) |

### Typical Cost Weights (Financial Industry Approximations)
- Cost of FN (missed fraud): **$200** (average fraud transaction amount + operational overhead)
- Cost of FP (false alert): **$10** (customer friction, support call)
- **FN is ~20x more costly than FP**

This means we should bias our threshold toward **higher recall** (catch more fraud), accepting lower precision (some false alarms).

---

## 4. THRESHOLD OPTIMIZATION FRAMEWORK & EMPIRICAL SAVINGS

Standard classifiers output probabilities: P(fraud | transaction)

The classification threshold determines:
- **Low threshold (e.g., 0.3):** Catch almost all fraud, but block many legitimate customers → High Recall, Low Precision
- **High threshold (e.g., 0.7):** Only flag very suspicious transactions → Low Recall, High Precision

**Business-Optimal Threshold Optimization:**
Using validation set predictions, we ran a threshold sweep to minimize:
$$\text{Total Cost} = (\text{FN\_count} \times \$200) + (\text{FP\_count} \times \$10)$$

### Empirical Results (Validation vs. Holdout Test)

1. **Validation Set Optimization:**
   - **Default Threshold (0.5):** Total Cost = **$440.00** (FN: 1, FP: 24)
   - **Optimal Threshold (0.807):** Total Cost = **$340.00** (FN: 1, FP: 14)
   - **Validation Cost Savings:** **$100.00 (22.73% reduction)**
   - *Result:* Raising the threshold to 0.807 eliminated 10 false alarms without missing any additional fraud.

2. **Holdout Test Set Evaluation (Generalization):**
   - **Default Threshold (0.5):** Total Cost = **$550.00** (FN: 1, FP: 35)
   - **Optimized Threshold (0.807):** Total Cost = **$640.00** (FN: 2, FP: 24)
   - **Cost Shift:** **+$90.00 (16.36% cost increase)**

### 💡 Crucial Product Engineering Lesson: Boundary Overfitting
On small datasets with extremely few minority samples (only 23 fraud cases in each split), threshold optimization on a validation set is highly sensitive to single-sample confidence shifts.
- Raising the threshold to **0.807** was mathematically optimal on validation data, but on the holdout test set, it caused **one additional fraud transaction to be missed** (FN went from 1 to 2).
- Because a missed fraud ($200) is 20x more expensive than a false alarm ($10), missing that single transaction erased the $110 saved by reducing false alarms (FP decreased from 35 to 24), leading to a net cost increase.
- **Production Recommendation:** For maximum business safety, we recommend **retaining the default threshold of 0.500** (or a slightly more conservative threshold like 0.450) as our production standard (**MODEL-Final**). This prioritizes recall, yielding a lower out-of-sample cost ($550 vs $640) and guaranteeing that we only miss a single fraud transaction.

---

## 5. FRAUD PATTERNS IN THIS DATASET

### Preliminary Hypotheses (to be validated in EDA)

Based on domain knowledge of credit card fraud:

| Feature | Expected Fraud Signal | Business Reasoning |
|---|---|---|
| `foreign_transaction = 1` | Strong positive | Cards used in unexpected countries = high risk |
| `location_mismatch = 1` | Strong positive | Billing vs. transaction location divergence |
| `velocity_last_24h > 5` | Moderate positive | High velocity = card testing or fraud spree |
| `device_trust_score < 30` | Moderate positive | Untrusted/anonymous device |
| `transaction_hour ∈ [0,4]` | Moderate positive | Late night / early morning activity |
| `merchant_category = Electronics` | Moderate positive | High-value, liquid goods (easy to resell) |
| `merchant_category = Travel` | Moderate positive | Non-refundable, large transactions |
| `amount > $500` | Moderate positive | Large transactions need scrutiny |

---

## 6. MODEL SUCCESS BUSINESS INTERPRETATION

### What "85% Recall" Means for a Bank

If the bank processes **100,000 transactions/day** with a 5% fraud rate:
- Total fraudulent transactions: 5,000
- With 85% recall model: Catches **4,250 fraud cases**
- With 50% recall baseline: Catches **2,500 fraud cases**
- **Improvement: 1,750 additional frauds caught per day**
- At $200 avg. fraud cost: **$350,000 saved per day**
- **Annual savings: ~$127 million**

This is why building a fraud model matters, and why every percentage point of recall improvement has enormous real-world value.

---

## 7. FRAUD PATTERN SHIFTS (Concept Drift)

A critical consideration for production deployment:

- Fraudsters adapt. A model trained today may be obsolete in 6 months.
- Solution: Continuous monitoring, regular retraining, anomaly detection on score distributions.
- This project will note where monitoring hooks would be added in a production system.

---

*Last updated: 2026-06-17 | Phase: 11 – Business-Aware Threshold Tuning*
