# Logistic Regression

## Table of Contents
1. What is Logistic Regression?
2. Why It Works (Sigmoid + Log-Odds)
3. Decision Boundary and Thresholding
4. Assumptions and When to Use
5. Regularization and Class Imbalance
6. Diagnostics and Pitfalls
7. Interview Insights

---

## 1) What is Logistic Regression?

A probabilistic classifier for binary outcomes. It models the log-odds of the positive class as a linear function of features and maps to probability via the sigmoid.

- Output: probability in [0, 1]
- Decision: classify by comparing probability to a threshold (default 0.5)

Use-cases: churn prediction, fraud detection, lead conversion, disease presence.

---

## 2) Why It Works (Sigmoid + Log-Odds)

- Linear score (logit): score = w₀ + w₁x₁ + … + wₙxₙ
- Log-odds: log(p/(1−p)) = score  
- Probability: p = 1 / (1 + e^(−score))

Interpretation: Each coefficient represents change in log-odds for a unit feature increase, holding others fixed; exponentiated coefficients give odds ratios.

Why sigmoid? Monotonic mapping to [0,1], differentiable, stable training.

---

## 3) Decision Boundary and Thresholding

- Decision boundary: set of points where p = 0.5 ⇒ score = 0 (a hyperplane)
- For imbalanced costs, choose threshold ≠ 0.5 to trade precision vs recall
- Calibrate probabilities if needed (Platt scaling, isotonic regression)

---

## 4) Assumptions and When to Use

Assumptions:
- Log-odds are a linear function of features
- Independence of observations
- Limited multicollinearity
- Large samples help asymptotic normality for inference

When to use:
- You need calibrated probabilities
- Baseline model for fast, interpretable deployment
- Features approximately linear in log-odds; with interactions/transformations, handles mild non-linearities

---

## 5) Regularization and Class Imbalance

Regularization:
- L2 (Ridge): stabilizes coefficients, handles multicollinearity
- L1 (Lasso): performs feature selection (sparse solution)
- Elastic Net: balance L1/L2 for correlated groups

Class imbalance:
- Adjust class weights (cost-sensitive learning)
- Calibrated thresholding using PR curves
- Use appropriate metrics (Precision/Recall, PR-AUC) not just accuracy
- Resampling is a last resort for LR; class weighting often sufficient

---

## 6) Diagnostics and Pitfalls

Diagnostics:
- ROC/PR curves for threshold selection
- Confusion matrix for precision/recall trade-offs
- Calibration curves to verify probability quality
- Residual analysis (deviance residuals) for outliers/influence

Pitfalls:
- Linear decision boundary may be too simple
- Poorly scaled features can slow/unstabilize training (with regularization)
- Leakage from future/target-derived features inflates performance
- Over-reliance on accuracy in imbalanced settings

---

## 7) Interview Insights

- Logistic vs Linear Regression: LR models log-odds and outputs probabilities; LR’s MSE-based approach is not suitable for classification.
- Threshold selection: driven by business costs (false positives vs false negatives).
- Why regularize? Prevent overfitting and improve generalization.
- Business angle: Easy to explain to stakeholders (odds ratios), quick to deploy, strong baseline for many classification tasks.
