# K-Means Clustering

## Table of Contents
1. Intuition and Objective
2. Algorithm Steps and Convergence
3. Initialization (k-means++)
4. Choosing K (Model Selection)
5. Strengths, Limitations, Pitfalls
6. Interview Insights

---

## 1) Intuition and Objective

K-Means partitions data into K clusters by assigning each point to the nearest centroid and updating centroids as the mean of assigned points. It minimizes within-cluster sum of squares (inertia).

Intuition:
- Centroid = prototype of a cluster
- Points pulled toward closest centroid
- Clusters are Voronoi regions around centroids

Objective (conceptual): minimize average squared distance of points to their cluster centroids.

---

## 2) Algorithm Steps and Convergence

Steps:
1. Initialize K centroids
2. Assignment: assign each point to closest centroid
3. Update: recompute centroids as mean of assigned points
4. Repeat 2–3 until assignments stabilize (converged) or max iterations reached

Convergence:
- Guaranteed to a local optimum of inertia
- Different initializations can yield different solutions → run multiple times and choose best inertia

---

## 3) Initialization (k-means++)

Problem: Random initialization can produce poor local minima.

Solution: k-means++ seeding
- Pick first centroid uniformly at random
- Subsequent centroids chosen with probability proportional to squared distance from nearest existing centroid
- Leads to better starting positions and faster convergence

---

## 4) Choosing K (Model Selection)

Elbow method:
- Plot inertia vs K
- Choose K at the "elbow" where marginal gain diminishes

Silhouette score:
- Measures cohesion vs separation in [−1, 1]
- Higher = better clustering

Domain knowledge:
- Business-driven segment count (e.g., 4 customer tiers)

Caveat: True structure may not be spherical or evenly sized; consider alternative algorithms.

---

## 5) Strengths, Limitations, Pitfalls

Strengths:
- Simple, fast, scalable (mini-batch variants)
- Works well for spherical, equally sized clusters
- Interpretability via centroids

Limitations:
- Assumes spherical clusters of similar size and density
- Sensitive to outliers (means shift)
- Requires K specified upfront
- Distance metric sensitive to scaling

Pitfalls:
- Not scaling features → distorted distances
- Using K-Means for non-globular clusters → poor segmentation
- Ignoring initialization variability → unstable results

---

## 6) Interview Insights

- When is K-Means appropriate? Numeric data with roughly spherical clusters and similar scales.
- How to handle outliers? Remove/robustify or use K-Medoids/DBSCAN.
- How to scale to large data? Mini-batch K-Means; approximate nearest centroids.
- Business angle: Quick, actionable segmentation with centroid profiles for marketing and operations.
