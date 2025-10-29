# Apriori Algorithm (Association Rule Mining)

## Table of Contents
1. Problem Setting
2. Core Ideas: Support, Confidence, Lift
3. Apriori Principle and Algorithm Intuition
4. Strengths, Limitations, Pitfalls
5. Interview Insights

---

## 1) Problem Setting

Given a set of transactions (e.g., market baskets), discover itemsets that frequently occur together and derive association rules (e.g., {bread, butter} → {jam}).

---

## 2) Core Ideas: Support, Confidence, Lift

- Support(X): fraction of transactions containing X (prevalence)
- Confidence(X→Y): support(X∪Y) / support(X) (conditional probability of Y given X)
- Lift(X→Y): confidence(X→Y) / support(Y) (how much X increases likelihood of Y)
  - Lift > 1 → positive association
  - Lift = 1 → independence
  - Lift < 1 → negative association

Quality control: filter rules by minimum support, confidence, and possibly lift.

---

## 3) Apriori Principle and Algorithm Intuition

Apriori principle: All subsets of a frequent itemset must be frequent. Contrapositive: if a subset is infrequent, supersets cannot be frequent.

Algorithm intuition:
1. Find frequent 1-itemsets (meet min support)
2. Generate candidate 2-itemsets from frequent 1-itemsets
3. Prune candidates whose subsets are not all frequent
4. Count supports and keep frequent ones
5. Repeat for k-itemsets until no candidates remain

Finally, generate rules from frequent itemsets that meet confidence/lift thresholds.

---

## 4) Strengths, Limitations, Pitfalls

Strengths:
- Simple, explainable rules for co-occurrence
- Useful for cross-selling, recommendation, layout optimization

Limitations:
- Combinatorial explosion at high dimensionality
- Requires multiple scans of the dataset
- Rules can be redundant; post-processing needed

Pitfalls:
- Setting thresholds too low → too many, noisy rules
- Ignoring lift → spurious rules from common items
- Not grouping items (taxonomies) → fragmented insights

---

## 5) Interview Insights

- When to use? Transactional data with meaningful co-occurrence (retail, web clicks, medical co-prescriptions)
- Why Apriori vs FP-Growth? Apriori is conceptually simple but slower; FP-Growth is memory-efficient and faster for large data.
- Business angle: Data-driven bundling, promotions, and store layout improvements; explains "why" items are recommended together.
