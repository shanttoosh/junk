# Bias–Variance Trade-off

## Table of Contents
1. Conceptual Decomposition
2. Practical Implications
3. Diagnosing and Managing Trade-offs
4. Interview Insights

---

## 1) Conceptual Decomposition

Expected prediction error ≈ Bias² + Variance + Irreducible noise.

- Bias: error from erroneous assumptions/restrictive model class
- Variance: sensitivity to training sample fluctuations
- Noise: inherent randomness in data generating process

Low bias models (complex) often have high variance; high bias models (simple) have low variance.

---

## 2) Practical Implications

- Increasing model complexity lowers bias but raises variance
- Regularization increases bias but lowers variance
- More data typically reduces variance without increasing bias

Examples:
- Linear regression: high bias if relationship non-linear
- Deep trees: low bias, high variance; random forests reduce variance by averaging

---

## 3) Diagnosing and Managing Trade-offs

Diagnose:
- Learning curves: train vs validation error
- Validation curves: metric vs model complexity

Manage:
- Regularization (L1/L2), ensembling, early stopping
- Feature engineering to reduce bias without increasing variance excessively
- Data augmentation or more samples to reduce variance

---

## 4) Interview Insights

- There’s no free lunch: choose a point on the curve based on business needs (accuracy vs stability)
- Explain how your design (e.g., random forest) targets variance reduction
- Business angle: Stable models with controlled variance deliver predictable performance in production
