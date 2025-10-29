# Unsupervised Learning

## Table of Contents
1. What is Unsupervised Learning?
2. Core Tasks (Clustering, Association, Dimensionality Reduction)
3. When to Use and Evaluation
4. Data and Preprocessing Considerations
5. Interview Insights

---

## 1) What is Unsupervised Learning?

Learning patterns from unlabeled data. The algorithm discovers structure without explicit target labels.

Key goals:
- Group similar items (clustering)
- Find co-occurrence patterns (association rules)
- Reduce dimensionality while retaining structure (PCA)

Use-cases: segmentation (customers, products), anomaly detection, recommendation, topic discovery.

---

## 2) Core Tasks

### Clustering
- Objective: partition points so members of a cluster are similar
- Algorithms: K-Means, Hierarchical, DBSCAN
- Similarity: distance metrics (Euclidean, cosine), linkage strategies

### Association Rule Mining
- Objective: discover items that frequently occur together in transactions
- Algorithms: Apriori, FP-Growth, ECLAT
- Outputs: rules of the form X → Y with support, confidence, lift

### Dimensionality Reduction
- Objective: compress features while preserving structure
- Methods: PCA (linear), t-SNE/UMAP (non-linear, visualization)

---

## 3) When to Use and Evaluation

When to use:
- No labels available
- Exploratory analysis to reveal groups or patterns
- Preprocessing for supervised tasks (features, noise reduction)

Evaluation (no labels):
- Internal metrics: silhouette score, Davies–Bouldin index (cohesion/separation)
- Stability: split data, compare cluster consistency
- Domain validation: business interpretability and actionability

---

## 4) Data and Preprocessing Considerations

- Scaling: critical for distance-based methods (K-Means, DBSCAN)
- Dimensionality: high-d degrades distances → consider PCA first
- Density/Shape: choose algorithm based on cluster shape (K-Means: spherical; DBSCAN: arbitrary)
- Noise/outliers: DBSCAN robust; K-Means sensitive
- Feature design: use meaningful features to induce useful groupings

---

## 5) Interview Insights

- Clustering vs classification: clustering has no labels; results require validation and interpretation.
- Why multiple algorithms? Different assumptions (centroid vs density vs hierarchy) capture different structures.
- How to choose K? Use elbow/silhouette and domain context; no universal answer.
- Business angle: segments enable targeted marketing, product personalization, and anomaly detection.
