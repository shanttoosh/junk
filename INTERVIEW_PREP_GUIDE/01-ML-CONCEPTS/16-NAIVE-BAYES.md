# Naive Bayes Classifiers

## Table of Contents
1. Intuition and Bayes Rule
2. Naive Independence Assumption
3. Variants and When to Use
4. Strengths, Limitations, Pitfalls
5. Interview Insights

---

## 1) Intuition and Bayes Rule

Goal: compute P(class | features) from data.

Bayes rule (conceptually): posterior ∝ likelihood × prior
- Prior: how probable a class is a priori
- Likelihood: how probable features are given the class
- Posterior: how probable the class is given the features

---

## 2) Naive Independence Assumption

Assume features are conditionally independent given the class:
- P(x₁, x₂, …, xₙ | y) = Π P(xᵢ | y)
- Simplifies learning to estimating per-feature likelihoods for each class

Despite being a strong assumption, works surprisingly well in practice (especially text).

---

## 3) Variants and When to Use

- Gaussian NB: continuous features modeled with Gaussian per class; fast baseline for numeric data
- Multinomial NB: counts/frequencies (e.g., word counts); widely used in NLP
- Bernoulli NB: binary features (word present/absent)

Use-cases:
- Text classification (spam, sentiment)
- Baselines for high-dimensional sparse data
- Problems where independence is not terribly violated

Handling zero probabilities:
- Laplace (add-one) smoothing prevents zero likelihood for unseen features

---

## 4) Strengths, Limitations, Pitfalls

Strengths:
- Extremely fast to train/predict
- Works well with many features and small data
- Robust with noisy features

Limitations:
- Independence assumption rarely true
- Continuous features not Gaussian → poor Gaussian NB fit
- Probability calibration sometimes poor

Pitfalls:
- Ignoring smoothing → zeros kill posterior
- Using accuracy on imbalanced data; prefer precision/recall/PR-AUC

---

## 5) Interview Insights

- Why NB for text? High-dimensional sparse features; independence is reasonable approximation; speed.
- Which variant when? Multinomial for counts; Bernoulli for binary presence; Gaussian for continuous.
- Business angle: Lightning-fast baseline for large-scale text filtering and real-time classification.
