# Cross Validation

## Table of Contents
1. Why Cross-Validate?
2. K-Fold and Stratified K-Fold
3. Nested CV and Model Selection
4. Time Series Cross-Validation
5. Leakage and Best Practices
6. Interview Insights

---

## 1) Why Cross-Validate?

- Estimate out-of-sample performance reliably
- Reduce variance of a single train/validation split
- Use data efficiently when dataset is not huge

---

## 2) K-Fold and Stratified K-Fold

K-Fold:
- Split data into K folds
- Train on K−1 folds, validate on the remaining fold
- Repeat K times; average metrics

Stratified K-Fold:
- Maintains class proportions in each fold
- Essential for classification with imbalance

Typical K: 5 or 10; larger K reduces bias but increases variance and compute.

---

## 3) Nested CV and Model Selection

Problem: Optimism if hyperparameters tuned on the same CV used for evaluation.

Nested CV:
- Outer loop: estimates generalization
- Inner loop: tunes hyperparameters

Use nested CV for fair model comparison when data is limited.

---

## 4) Time Series Cross-Validation

Random K-Fold invalid for temporal data (leakage from future to past).

Approach:
- Expanding window or rolling window validation
- Train on past, validate on future; move window forward in time

Respects temporal ordering and concept drift.

---

## 5) Leakage and Best Practices

- Apply preprocessing (scaling, encoding, imputation) inside CV folds only
- Don’t peek at validation data during feature engineering or tuning
- Use pipelines to ensure transformations are fit only on training folds
- For rare events, ensure folds are stratified and sufficiently populated

---

## 6) Interview Insights

- Why CV over a single split? Less variance and better use of data
- When not to CV? Extremely large data where a holdout suffices; streaming systems
- Business angle: Provides trustworthy performance estimates before deployment, reducing risk
