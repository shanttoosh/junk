# Regularization in Machine Learning

## Table of Contents
1. Why Regularize?
2. L2 (Ridge), L1 (Lasso), Elastic Net
3. Early Stopping and Capacity Control
4. Dropout (Conceptual) and Noise Injection
5. When to Use and Trade-offs
6. Interview Insights

---

## 1) Why Regularize?

- Prevent overfitting by discouraging overly complex models
- Improve generalization by constraining parameter magnitude or capacity
- Stabilize solutions in presence of multicollinearity and noise

---

## 2) L2 (Ridge), L1 (Lasso), Elastic Net

L2 (Ridge): penalizes squared magnitude of weights
- Shrinks coefficients smoothly toward zero
- Handles multicollinearity; never sets coefficients exactly to zero

L1 (Lasso): penalizes absolute magnitude of weights
- Creates sparsity by driving some coefficients to zero
- Implicit feature selection; can pick one among correlated features

Elastic Net: combination of L1 and L2
- Balances sparsity and stability
- Useful with groups of correlated features; tends to keep groups

Tuning:
- Regularization strength (λ) controls bias–variance trade-off
- Standardize features to make penalty fair across dimensions

---

## 3) Early Stopping and Capacity Control

Early stopping:
- Monitor validation loss; stop training when it stops improving
- Prevents overfitting in iterative learners (GBMs, neural nets)

Capacity control:
- Limit tree depth/leaves in decision trees/GBMs
- Reduce number of basis functions or hidden units

---

## 4) Dropout (Conceptual) and Noise Injection

Dropout (NNs): randomly deactivate a fraction of units during training
- Prevents co-adaptation; acts like an ensemble of subnetworks
- At inference, use full network with scaled weights

Noise injection:
- Add noise to inputs/weights or labels (carefully) to improve robustness

---

## 5) When to Use and Trade-offs

Use when:
- Overfitting observed (train ≫ validation performance)
- High-dimensional features; multicollinearity present
- Need simpler, more interpretable models (L1)

Trade-offs:
- Too much regularization → underfitting (high bias)
- Too little → overfitting (high variance)

---

## 6) Interview Insights

- L1 vs L2: sparsity vs stability; Elastic Net for correlated groups
- Why standardize? To ensure penalty treats features equally
- Early stopping: a strong, practical regularizer widely used in boosting/NNs
- Business angle: Regularization yields reliable, stable models under real-world drift and noise
