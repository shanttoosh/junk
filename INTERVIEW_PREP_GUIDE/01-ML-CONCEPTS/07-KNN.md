# K-Nearest Neighbors (KNN)

## Table of Contents
1. What is KNN?
2. Distance Metrics and Scaling
3. Choosing K and Bias–Variance
4. Weighted KNN and Variants
5. Strengths, Limitations, Pitfalls
6. Interview Insights

---

## 1) What is KNN?

A non-parametric, instance-based method that classifies/regresses a query point by aggregating the labels/values of its K closest training samples.

- Classification: majority vote among K neighbors
- Regression: average of K neighbors
- No explicit training; prediction requires full dataset

Use cases: recommendation baselines, anomaly detection, small datasets with local structure.

---

## 2) Distance Metrics and Scaling

Common distances:
- Euclidean (L2): √Σ (xᵢ − yᵢ)²
- Manhattan (L1): Σ |xᵢ − yᵢ|
- Cosine similarity: angle-based, useful for high-dimensional sparse vectors
- Hamming: for categorical/binary features

Scaling matters:
- Features on larger scales dominate distances
- Standardize/normalize features prior to KNN
- Mixed data types: use appropriate distance (e.g., Gower) or encode carefully

---

## 3) Choosing K and Bias–Variance

- Small K (e.g., 1–3): low bias, high variance (sensitive to noise)
- Large K: higher bias, lower variance (smoother boundaries)
- Odd K prevents ties in binary classification

Model selection:
- Choose K via cross-validation optimizing the target metric (e.g., F1 for imbalance)
- Consider different distance metrics; sometimes cosine outperforms Euclidean

---

## 4) Weighted KNN and Variants

Weighted KNN:
- Weight neighbors by inverse distance (closer neighbors influence more)
- Often improves performance over uniform weights

Variants:
- Radius neighbors: all points within a radius r (varying K)
- Approximate nearest neighbors for speed (LSH, HNSW)

---

## 5) Strengths, Limitations, Pitfalls

Strengths:
- Simple, intuitive, strong non-linear baseline
- Naturally multi-class
- Works well when decision boundary is locally smooth

Limitations:
- Prediction can be slow (O(N) per query); needs indexing/ANN to scale
- Memory heavy (store full dataset)
- Curse of dimensionality: distances lose meaning in high dimensions

Pitfalls:
- Ignoring feature scaling
- Using Euclidean on sparse/high-d data where cosine is better
- Using accuracy on imbalanced datasets; prefer F1/PR-AUC

---

## 6) Interview Insights

- When is KNN appropriate? Small to medium datasets with clear local structure and moderate dimensionality.
- How to speed it up? KD-Trees/Ball Trees (low dimensions), approximate methods (HNSW/FAISS) for high dimensions.
- Business angle: Quick, explainable baseline; good for prototyping recommender similarity and anomaly detection.
