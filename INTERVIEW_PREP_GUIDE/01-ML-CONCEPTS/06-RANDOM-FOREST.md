# Random Forest

## Table of Contents
1. What is a Random Forest?
2. Why It Works (Variance Reduction)
3. Key Hyperparameters
4. Feature Importance and OOB Error
5. Strengths, Limitations, Pitfalls
6. Interview Insights

---

## 1) What is a Random Forest?

An ensemble of decision trees trained on bootstrapped samples of the data with random feature selection at each split. Predictions are aggregated (majority vote for classification, average for regression).

Intuition:
- Each tree overfits differently
- Aggregating many overfit trees cancels out errors
- Result: lower variance, better generalization

---

## 2) Why It Works (Variance Reduction)

Single trees have high variance. By injecting randomness (bagging + feature subsampling), trees become less correlated. Averaging many weakly correlated predictors reduces variance substantially (law of large numbers), improving stability and accuracy.

Correlated trees bring less benefit; feature subsampling decorrelates them.

---

## 3) Key Hyperparameters

- n_estimators: number of trees (100–1000 typical). More → better up to a point.
- max_depth: controls tree depth; deeper → lower bias, higher variance.
- max_features:
  - Classification: sqrt(#features) is common
  - Regression: #features/3 is common
  - Lower values → more decorrelation, sometimes better generalization
- min_samples_split / min_samples_leaf: regularize leaves and prevent tiny regions
- class_weight: handle imbalance without resampling

---

## 4) Feature Importance and OOB Error

Feature importance:
- Impurity-based importance (built-in) can be biased
- Permutation importance (preferred) measures drop in performance when a feature is shuffled

Out-of-bag (OOB) error:
- ~37% of samples are left out of each tree’s bootstrap sample
- Use OOB samples as a built-in validation set to estimate generalization without separate CV

---

## 5) Strengths, Limitations, Pitfalls

Strengths:
- Strong baseline with minimal tuning
- Robust to noise and outliers
- Handles mixed data types and non-linearities
- Inherent feature ranking via importance

Limitations:
- Less interpretable than a single tree
- Large models can be memory-intensive
- May be slow for very high-dimensional sparse data

Pitfalls:
- Relying solely on impurity importance (biased to high-cardinality)
- Not adjusting for class imbalance (use class_weight or calibrated thresholds)
- Using too shallow trees → high bias

---

## 6) Interview Insights

- Why Random Forest over a single tree? Lower variance, better generalization.
- How to handle imbalance? class_weight, thresholding, or stratified sampling.
- When to favor Gradient Boosting? When you need higher accuracy and can afford tuning complexity; RF is stronger as a robust baseline.
- Business angle: Reliable, fast to deploy, little preprocessing, strong results across domains.
