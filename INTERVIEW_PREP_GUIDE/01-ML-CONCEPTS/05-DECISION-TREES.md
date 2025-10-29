# Decision Trees

## Table of Contents
1. What is a Decision Tree?
2. Splitting Criteria (Impurity Measures)
3. Stopping and Pruning
4. Interpretability and Feature Importance
5. Strengths and Limitations
6. Interview Insights

---

## 1) What is a Decision Tree?

A tree-structured model that recursively splits the feature space into regions with increasingly homogeneous target values. Leaves represent predictions; internal nodes represent decision rules.

High-level flow:
```
Start with all data
 └─ Choose the best feature and split point (max impurity reduction)
    ├─ Left subset → repeat
    └─ Right subset → repeat
Stop when a stopping criterion is met → assign leaf prediction
```

---

## 2) Splitting Criteria (Impurity Measures)

For classification:
- Gini Impurity: 1 − Σ pᵢ² (lower is purer)
- Entropy: −Σ pᵢ log₂ pᵢ (information-theoretic purity)
- Information Gain: Parent impurity − weighted sum of child impurities

For regression:
- Variance Reduction: Parent variance − weighted child variances

Choosing splits:
- For numeric features: evaluate candidate thresholds
- For categorical features: evaluate subsets/one-vs-rest splits

Biases:
- Features with many distinct values may appear attractive; use constraints to mitigate (min samples per split, max depth).

---

## 3) Stopping and Pruning

Stopping criteria:
- Maximum depth
- Minimum samples per split/leaf
- Minimum impurity decrease

Pruning (to combat overfitting):
- Pre-pruning: apply limits during growth (depth, min samples)
- Post-pruning: grow full tree, then prune back using validation (cost-complexity pruning)

Visual cue of overfitting:
- Training accuracy ≫ validation accuracy; deep tree memorizes noise

---

## 4) Interpretability and Feature Importance

Interpretability:
- Path-based explanations: "If contract = month-to-month and tenure < 6 then churn likely"
- Global structure: small trees are human-readable; large trees are not

Feature importance (tree-based):
- Importance = total impurity reduction credited to a feature across the tree
- Caution: Biased toward high-cardinality and continuous features; use permutation importance for fairer assessment

---

## 5) Strengths and Limitations

Strengths:
- Non-linear, captures interactions naturally
- Handles numeric and categorical data
- Little preprocessing required (no scaling needed)
- Fast inference and easy to visualize (when shallow)

Limitations:
- High variance (unstable w.r.t. data perturbations)
- Overfitting prone without pruning
- Axis-aligned splits may struggle with certain boundaries
- Bias toward features with many levels without constraints

When to use:
- Need interpretability and non-linear modeling
- As base learners in ensembles (Random Forest, Gradient Boosting)

---

## 6) Interview Insights

- Entropy vs Gini: Often similar splits; Gini is slightly faster; entropy has information-theoretic grounding.
- Why prune? Controls variance and improves generalization.
- How to handle categorical variables? One-hot or specialized categorical handling; beware high-cardinality.
- Business angle: Produces rule sets that stakeholders understand; good for policy/eligibility decisions.
