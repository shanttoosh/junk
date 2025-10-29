# Hierarchical Clustering

## Table of Contents
1. Agglomerative vs Divisive
2. Linkage Criteria
3. Dendrograms and Cutting the Tree
4. Distance Metrics and Scaling
5. Strengths, Limitations, Pitfalls
6. Interview Insights

---

## 1) Agglomerative vs Divisive

- Agglomerative (bottom-up): start with each point as its own cluster, iteratively merge the closest clusters until one cluster remains.
- Divisive (top-down): start with all points in one cluster, iteratively split into smaller clusters.

Agglomerative is more common due to simpler computation and available linkage schemes.

---

## 2) Linkage Criteria

How to define distance between clusters:
- Single linkage: min pairwise distance (can chain clusters → elongated shapes)
- Complete linkage: max pairwise distance (compact, tight clusters)
- Average linkage: average pairwise distance (balanced)
- Ward’s method: merges that minimize increase in total within-cluster variance (often gives compact, spherical clusters)

Choice affects cluster shape and robustness to noise.

---

## 3) Dendrograms and Cutting the Tree

Dendrogram: a tree that visualizes the sequence of merges.

How to decide cluster count:
- Look for "long vertical lines" (large distances at which merges happen)
- Cut the tree at a height before big jumps

Advantages:
- Provides a full hierarchy; explore multiple granularities of segments
- No need to pre-specify K

---

## 4) Distance Metrics and Scaling

- Distance: Euclidean is common; cosine for high-dimensional sparse data
- Scaling: standardize features to prevent dominance by large-scale features
- Dimensionality reduction (PCA) can help stability and speed

---

## 5) Strengths, Limitations, Pitfalls

Strengths:
- No K required upfront; dendrogram reveals structure
- Captures nested clusters and various shapes depending on linkage
- Good for exploratory analysis

Limitations:
- O(n² log n) to O(n³) complexity; not ideal for very large datasets
- Sensitive to noise/outliers (especially single linkage)
- Results can vary with metric/linkage choices

Pitfalls:
- Interpreting dendrogram without domain knowledge
- Choosing linkage that doesn’t match cluster geometry
- Not scaling features → distorted distances

---

## 6) Interview Insights

- When to use? Small-to-medium datasets where hierarchical relationships matter or K is unknown.
- Linkage trade-offs: single (chaining), complete (tight clusters), Ward (variance minimization).
- Business angle: Delivers multi-level customer segments for different campaign granularity; easy to explain with dendrograms.
