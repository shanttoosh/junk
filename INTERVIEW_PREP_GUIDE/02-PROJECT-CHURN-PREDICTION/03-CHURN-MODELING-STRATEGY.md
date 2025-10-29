# Churn Prediction — Modeling Strategy

## Table of Contents
1. Candidate Algorithms and Rationale
2. Model Selection Criteria
3. Thresholding Strategy (Cost-Aware)
4. Probability Calibration
5. Risk Segmentation and Actions
6. Governance and Reproducibility

---

## 1) Candidate Algorithms and Rationale

Models considered (tabular classification):
- Logistic Regression (baseline, interpretability)
- Decision Tree (rule-based baseline)
- Random Forest (variance reduction, robust)
- Gradient Boosting (XGBoost/LightGBM/CatBoost) — strong performance on tabular
- SVM (margin-based baseline for small/medium datasets)
- Naive Bayes (fast, high-dimensional text-like scenarios)

Rationale:
- Start simple to set a baseline and interpret relationships
- Move to ensembles (RF/GBM) for accuracy and non-linear interactions
- CatBoost handles categorical variables natively; GBMs excel on mixed features

Expected outcome:
- GBMs typically outperform others on structured churn data, with RF close behind; LR remains valuable for interpretability and as a challenger model.

---

## 2) Model Selection Criteria

Technical metrics:
- Primary: PR-AUC (reflecting minority class focus)
- Secondary: ROC-AUC, F1, Recall@Top-K (operational)
- Stability: Std. dev. across CV folds; sensitivity to shifts
- Calibration: reliability curves, Brier score

Operational criteria:
- Inference latency and resource usage
- Ease of monitoring and retraining
- Interpretability requirements for marketing/legal

Decision framework:
- Shortlist top-2 based on PR-AUC and stability
- Choose final by business constraints (latency, interpretability) and calibration quality

---

## 3) Thresholding Strategy (Cost-Aware)

Set probability cutoff using cost matrix:
- Cost(FP): retention offer cost (discount, outreach)
- Cost(FN): lost LTV when true churn is missed

Minimize expected cost:
```
ExpectedCost(τ) = FP(τ) * Cost(FP) + FN(τ) * Cost(FN)
```

Practical approach:
- Plot precision/recall/cost vs threshold
- Choose τ to meet budget and capacity constraints (e.g., top 10% highest risk)
- Revisit quarterly as costs and acceptance rates change

---

## 4) Probability Calibration

Why calibrate:
- Marketing decisions depend on accurate risk estimates (not just ranking)

Methods:
- Platt scaling (logistic on scores)
- Isotonic regression (non-parametric, reliable with enough data)

Protocol:
- Calibrate on validation predictions; evaluate with calibration plots and Brier
- Monitor drift; re-calibrate if reliability worsens

---

## 5) Risk Segmentation and Actions

Segment by predicted risk for targeted actions:

Example tiers:
- Tier 1 (High risk: p ≥ 0.7): retention offer + agent outreach
- Tier 2 (Medium risk: 0.5 ≤ p < 0.7): personalized discount/upsell
- Tier 3 (Low risk but high value): proactive engagement (CS outreach)
- Tier 4 (Low risk, low value): nurture campaigns only

Action framework:
- Playbooks per tier (offer amount, message, channel)
- A/B test offers within tiers; optimize acceptance and net revenue

---

## 6) Governance and Reproducibility

- Version control for data, features, model artifacts, thresholds
- Model cards: document metrics, fairness checks, calibration status
- Audit trail: retain training configs, seeds, CV splits
- Challenger–champion setup: periodically evaluate challenger models (e.g., newer GBM) against champion in shadow/canary
