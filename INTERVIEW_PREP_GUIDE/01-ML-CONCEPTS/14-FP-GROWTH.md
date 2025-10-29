# FP-Growth (Frequent Pattern Growth)

## Table of Contents
1. Motivation and Overview
2. FP-Tree Structure and Compression
3. Mining Frequent Itemsets (Pattern Growth)
4. Comparison with Apriori
5. Interview Insights

---

## 1) Motivation and Overview

FP-Growth discovers frequent itemsets without candidate generation. It compresses the dataset into an FP-tree, then mines patterns directly, typically faster and more scalable than Apriori for large datasets.

---

## 2) FP-Tree Structure and Compression

- Sort items in each transaction by global frequency (descending)
- Insert ordered transactions into a prefix tree (FP-tree)
- Shared prefixes create compact representation (compression)
- Maintain header table linking nodes of the same item for traversal

Benefit: Avoids scanning the database repeatedly for each candidate size.

---

## 3) Mining Frequent Itemsets (Pattern Growth)

- For each item (bottom-up by frequency):
  1. Extract its conditional pattern base (paths leading to the item)
  2. Build a conditional FP-tree
  3. Recursively mine frequent itemsets from the conditional tree
- Concatenate item with frequent patterns discovered in the conditional tree

This "divide-and-conquer" mines all frequent itemsets efficiently.

---

## 4) Comparison with Apriori

- Apriori: generates and tests many candidate itemsets; multiple full DB scans
- FP-Growth: builds compact FP-tree; mines patterns without explicit candidates
- FP-Growth is generally faster on large, dense datasets; memory usage depends on dataset compressibility

When FP-Growth shines:
- Large datasets with many overlapping transactions
- Low support thresholds that would explode Apriori candidates

---

## 5) Interview Insights

- Why FP-Growth? Efficiency via compression and conditional mining; avoids candidate blow-up
- When might Apriori suffice? Small datasets or when conceptual simplicity is preferred
- Business angle: Scalable discovery of product affinities for recommendations and promotions
