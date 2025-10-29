# ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal)

## Table of Contents
1. Motivation and Vertical Data Format
2. Tidset Intersections for Support Counting
3. Search Strategy and Pruning
4. Comparison with Apriori / FP-Growth
5. Interview Insights

---

## 1) Motivation and Vertical Data Format

ECLAT mines frequent itemsets using a vertical representation:
- For each item, store the list of transaction IDs (tidset) containing it
- Example: A: {1,3,4,7}, B: {2,3,4}

Support of an itemset = size of the intersection of its members’ tidsets.

---

## 2) Tidset Intersections for Support Counting

To compute support of {A,B}:
- Intersect A’s tidset with B’s tidset: {1,3,4,7} ∩ {2,3,4} = {3,4}
- Support = |{3,4}| / total transactions

Extends recursively: {A,B,C} support = (A∩B) ∩ C.

Advantage: Counting reduces to fast set intersections, particularly efficient with sorted bitsets.

---

## 3) Search Strategy and Pruning

- Depth-first traversal of itemset lattice (bottom-up)
- Use support threshold to prune branches early (anti-monotonicity)
- Equivalence classes: group itemsets sharing prefixes to structure the search

---

## 4) Comparison with Apriori / FP-Growth

- Apriori: candidate generation with horizontal scans; can be slow
- FP-Growth: tree compression and conditional mining
- ECLAT: vertical format, efficient intersections; memory usage depends on tidset representation

When ECLAT fits:
- Sparse datasets where vertical representation is compact
- Intersection operations are cheap with bitmaps

---

## 5) Interview Insights

- Why vertical format? Turns support counting into intersections; efficient with bitset operations
- When to prefer ECLAT? Sparse data, many items, when vertical compression yields gains
- Business angle: Fast association discovery for large catalogs (retail, media bundles)
