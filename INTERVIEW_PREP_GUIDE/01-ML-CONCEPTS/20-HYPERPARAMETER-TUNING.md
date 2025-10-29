yes goon# Hyperparameter Tuning

## Table of Contents
1. Why Tuning Matters
2. Search Strategies
3. Early Stopping and Resource Allocation
4. Practical Tips and Pitfalls
5. Interview Insights

---

## 1) Why Tuning Matters

- Hyperparameters control model complexity and generalization
- Proper tuning can close large performance gaps
- Avoid overfitting the validation set by disciplined procedures

---

## 2) Search Strategies

Grid Search:
- Exhaustive combinations on a predefined grid
- Simple, parallelizable; inefficient in high dimensions

Random Search:
- Sample hyperparameters from distributions
- More efficient exploration; often better than grid under time constraints

Bayesian Optimization:
- Model performance as a function of hyperparameters (surrogate)
- Choose next trials to balance exploration/exploitation
- Efficient for expensive models (GBMs, deep nets)

Successive Halving / Hyperband:
- Allocate small budgets to many configurations, keep the best
- Efficiently prunes poor configs early

---

## 3) Early Stopping and Resource Allocation

- Use early stopping criteria to cut training when validation stops improving
- Allocate time budget per model; prioritize promising candidates
- Multi-fidelity tuning (fewer trees/epochs early) speeds selection

---

## 4) Practical Tips and Pitfalls

- Define metric aligned with business goal
- Log-scale sampling for parameters spanning orders of magnitude (e.g., learning rate)
- Use nested CV for unbiased comparison
- Guard against leakage; ensure folds are proper
- Re-tune when data distribution changes

Pitfalls:
- Overfitting to validation set from repeated runs
- Searching overly narrow ranges based on defaults
- Ignoring interactions between hyperparameters

---

## 5) Interview Insights

- Why random over grid? Better coverage with fewer evaluations
- When Bayesian? Expensive evaluations, small budgets
- Business angle: Tuning maximizes ROI from existing data/models without increasing model complexity or cost
