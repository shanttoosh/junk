# Support Vector Machine (SVM)

## Table of Contents
1. Intuition and Maximum Margin
2. Hard vs Soft Margin
3. Kernels (Non-linear SVM)
4. Hyperparameters and Scaling
5. Strengths, Limitations, Pitfalls
6. Interview Insights

---

## 1) Intuition and Maximum Margin

SVM finds the separating hyperplane that maximizes the margin between classes. The margin is the distance from the hyperplane to the nearest points (support vectors). Maximizing this margin improves generalization.

Geometric picture:
```
Class +         |  ↑ margin  |        Class −
  o   o      o  |<———gap———>|  x   x      x
      o   o     |  hyperplane|     x   x
----------------+------------+----------------
      support vectors define the margin
```

Key idea:
- Only a subset of points (support vectors) determine the decision boundary.
- Other points can move without changing the hyperplane.

---

## 2) Hard vs Soft Margin

- Hard margin: No misclassifications; feasible only when perfectly separable and noise-free; overfits in practice.
- Soft margin: Allows some violations via slack variables, controlled by C.
  - Large C: penalizes misclassifications heavily → narrow margin (low bias, high variance)
  - Small C: tolerates violations → wider margin (higher bias, lower variance)

Trade-off: Fit training data vs margin width (bias–variance balance).

---

## 3) Kernels (Non-linear SVM)

When data is not linearly separable, SVM uses kernels to implicitly map data into a higher-dimensional space where a linear separator exists.

Common kernels:
- Linear: fast, baseline for high-dimensional sparse data (text)
- Polynomial: captures interactions up to degree d; can overfit at high degrees
- RBF (Gaussian): very flexible, default choice; γ controls locality of influence

Kernel trick: Compute inner products in feature space without explicit mapping, enabling efficient non-linear boundaries.

Choosing kernel:
- Text/high-dimensional sparse → linear
- Generic tabular with non-linear boundaries → RBF
- Known polynomial interactions → polynomial

---

## 4) Hyperparameters and Scaling

- C (regularization strength): controls margin violations
- γ (RBF): controls how far the influence of a single training example reaches
  - Large γ: very local, wiggly boundary (overfitting risk)
  - Small γ: smoother boundary (underfitting risk)
- Degree (polynomial): model complexity increases with degree

Scaling:
- Required. SVM optimization relies on distances/inner products.
- Standardize features for stable, fair influence across dimensions.

Class imbalance:
- Use class weights to penalize minority errors more strongly
- Evaluate with PR-AUC/F1; adjust the decision threshold accordingly

---

## 5) Strengths, Limitations, Pitfalls

Strengths:
- Effective in high dimensions; robust to overfitting with proper C/γ
- Works with different kernels; flexible decision boundaries
- Only support vectors matter → memory-efficient models

Limitations:
- Training can be slow on very large datasets (O(N²) or worse)
- Needs scaling; sensitive to hyperparameters
- Probabilities not directly produced (need calibration)

Pitfalls:
- Using RBF without scaling → unstable/poor results
- Default C/γ can severely under/overfit; tune via CV
- Interpreting SVM as inherently probabilistic; requires calibration for probabilities

---

## 6) Interview Insights

- Why SVM? Margin maximization yields good generalization, especially in high-dimensional spaces.
- Linear vs kernel SVM? Linear for text/high-d sparse; kernel for complex boundaries.
- How to tune? Grid/random search over C and γ (and degree for polynomial); evaluate with appropriate metrics.
- Business angle: Strong performance on moderate-sized datasets with clear margins; good for baseline and sensitive domains (spam filtering, text categorization).
