# Ensemble Learning

## Table of Contents
1. Why Ensembles Work
2. Bagging (Bootstrap Aggregating)
3. Boosting (Additive Models)
4. Stacking (Meta-Learning)
5. When to Use and Trade-offs
6. Interview Insights

---

## 1) Why Ensembles Work

Combine multiple diverse models to reduce errors:
- Variance reduction: averaging decorrelated models (bagging)
- Bias reduction: sequentially correct mistakes (boosting)
- Model combination: leverage complementary strengths (stacking)

Diversity is key: identical models make identical errors; decorrelation improves gains.

---

## 2) Bagging (Bootstrap Aggregating)

Mechanism:
- Sample with replacement to create multiple bootstrap datasets
- Train base learners (e.g., deep decision trees)
- Aggregate predictions (vote/average)

Examples: Random Forest (bagged trees + random feature selection)

Pros:
- Reduces variance; robust to overfitting
- Parallelizable; strong baseline

Cons:
- Larger memory/compute footprint
- Less interpretable than single model

---

## 3) Boosting (Additive Models)

Mechanism:
- Train weak learners sequentially; each focuses on previous errors
- Combine learners with weights to form strong model

Examples: AdaBoost, Gradient Boosting (GBM), XGBoost, LightGBM, CatBoost

Pros:
- High accuracy; handles mixed feature types
- Flexible loss functions; well-suited to tabular data

Cons:
- More tuning; risk of overfitting without regularization
- Sequential (less parallelism) and can be compute-intensive

Regularization in GBMs:
- Learning rate (shrinkage), tree depth, subsampling, L1/L2 on leaves
- Early stopping on validation

---

## 4) Stacking (Meta-Learning)

Mechanism:
- Train multiple base models (level-0)
- Use their out-of-fold predictions as features for a meta-model (level-1)
- Meta-model learns how to combine base outputs

Pros:
- Exploits complementary strengths
- Often yields strong performance in competitions

Cons:
- Complex, risk of leakage; requires careful OOF protocol
- Harder to interpret and maintain

---

## 5) When to Use and Trade-offs

Use:
- When single models underperform; need robustness or extra accuracy
- Tabular data: GBMs often state-of-the-art
- High variance base learners (trees) benefit from bagging

Trade-offs:
- Accuracy vs simplicity (interpretability, latency, cost)
- Maintenance complexity and reproducibility

---

## 6) Interview Insights

- Why ensembles? Leverage diversity to reduce error components
- Bagging vs boosting: variance vs bias reduction
- Stacking best practices: OOF predictions to avoid leakage; simple meta-models (e.g., LR)
- Business angle: Controlled accuracy gains with known costs; GBMs are strong production choices for many tabular problems
