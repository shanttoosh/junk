# Semi-Supervised Learning (SSL)

## Table of Contents
1. Motivation and Assumptions
2. Classic Approaches
3. Modern Approaches
4. When to Use and Risks
5. Interview Insights

---

## 1) Motivation and Assumptions

Motivation: Labels are expensive; unlabeled data is abundant. Use both to improve performance.

Common assumptions:
- Smoothness: similar points likely share labels
- Cluster: points in same cluster likely share labels
- Manifold: data lies on a low-dimensional manifold; decision boundaries should avoid high-density regions

---

## 2) Classic Approaches

Self-training:
- Train on labeled data; predict on unlabeled; add high-confidence predictions to labeled set; iterate
- Risk: confirmation bias; use confidence thresholds and regularization

Co-training:
- Two (or more) complementary views (feature sets) train models that teach each other
- Works when views are conditionally independent given label

Graph-based methods:
- Build a graph of samples; propagate labels across edges (similarity)

---

## 3) Modern Approaches

Consistency regularization:
- Encourage model to produce consistent predictions under input perturbations (augmentations)
- Π-model, Mean Teacher, FixMatch

Pseudo-labeling at scale:
- Use model (or teacher) to generate labels for unlabeled data; train student model on them
- Combined with strong data augmentation

---

## 4) When to Use and Risks

Use when:
- Limited labeled data; abundant unlabeled data from the same distribution
- Unlabeled data quality and domain match are high

Risks:
- Distribution shift between labeled and unlabeled reduces benefit
- Reinforcing errors via self-training; require confidence thresholds and validation

---

## 5) Interview Insights

- Why SSL? Leverage cheap unlabeled data to boost performance where labeling is costly
- Key assumption? Unlabeled data follows same distribution and structure as labeled
- Business angle: Reduces annotation cost while achieving near-supervised performance in many domains (vision, NLP)
