# Underfitting and Overfitting

## Table of Contents
1. Concepts and Intuition
2. Signs and Diagnostics
3. Remedies and Trade-offs
4. Bias–Variance Connection
5. Interview Insights

---

## 1) Concepts and Intuition

Underfitting: model too simple to capture patterns (high bias)  
Overfitting: model too complex, memorizes noise (high variance)

ASCII intuition:
```
True function:         Underfit:             Overfit:
  ·····█·····            ┌─────┐             ┌─┐ ┌┐┌─┐
  ··███▀██·            ──┘     └──         ──┘ └─┘└─┘
  ·█▀   ▀█·
```

---

## 2) Signs and Diagnostics

Underfitting:
- High training error and high validation error
- Learning curves plateau at high error even with more data

Overfitting:
- Low training error, high validation error
- Validation error increases as model complexity increases
- Large gap between train and validation performance

---

## 3) Remedies and Trade-offs

Fix underfitting:
- Increase model capacity (features, interactions, deeper trees)
- Reduce regularization
- Improve feature engineering

Fix overfitting:
- Add regularization (L1/L2), early stopping
- Reduce model complexity (prune trees, fewer parameters)
- More data / data augmentation
- Cross-validation and proper tuning

Trade-off: find the "sweet spot" of complexity that minimizes validation error.

---

## 4) Bias–Variance Connection

- Underfitting ↔ high bias, low variance
- Overfitting ↔ low bias, high variance
- Goal: minimize total error (bias² + variance + noise)

---

## 5) Interview Insights

- Use learning curves to diagnose; decide whether to add data or adjust complexity
- Proper CV central to detecting overfitting
- Business angle: Overfit models fail in production; robust models reduce risk and maintenance cost
