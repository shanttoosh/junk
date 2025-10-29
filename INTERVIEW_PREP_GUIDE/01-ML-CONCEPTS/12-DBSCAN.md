# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Table of Contents
1. Intuition and Definitions
2. Parameters: eps and min_samples
3. Algorithm Steps
4. Strengths, Limitations, Pitfalls
5. Interview Insights

---

## 1) Intuition and Definitions

DBSCAN groups points that are closely packed together (high density) and marks points in low-density regions as outliers.

Point categories:
- Core point: at least `min_samples` points within distance `eps`
- Border point: not core, but within `eps` of a core point
- Noise point: neither core nor border

Clustering emerges by expanding from core points and absorbing reachable points.

---

## 2) Parameters: eps and min_samples

- `eps` (ε): neighborhood radius; too small → many noise points; too large → merge clusters
- `min_samples`: minimum points to form dense region; typical heuristic: `min_samples ≈ dim + 1` or 4–10

Choosing `eps`:
- k-distance plot: sort distances to k-th nearest neighbor; look for the "knee" point as ε

Scaling:
- Required; distance-based method
- PCA can help in high dimensions to make density meaningful

---

## 3) Algorithm Steps

1. For each unvisited point, check if core (≥ `min_samples` neighbors within `eps`)
2. If core, start a new cluster and iteratively add density-reachable points
3. If not core and not density-reachable, label as noise (may later become border)

Result: Clusters of arbitrary shape with noise isolated.

---

## 4) Strengths, Limitations, Pitfalls

Strengths:
- Finds arbitrary-shaped clusters
- Robust to outliers (labels them as noise)
- No need to specify K

Limitations:
- Single global `eps` struggles with varying densities
- High-dimensional data: distances become less meaningful
- Parameter sensitivity; requires tuning

Pitfalls:
- Using Euclidean distance on non-isotropic features; standardize or use appropriate metrics
- Ignoring domain scale; `eps` too large/small ruins clustering
- Expecting crisp clusters for non-dense data

---

## 5) Interview Insights

- When to use DBSCAN? Irregular cluster shapes, presence of noise/outliers, unknown K.
- How to tune? Use k-distance plot for `eps`, heuristics for `min_samples`, scale features, consider PCA.
- Business angle: Good for anomaly detection and spatial/customer density segmentation without pre-setting K.
