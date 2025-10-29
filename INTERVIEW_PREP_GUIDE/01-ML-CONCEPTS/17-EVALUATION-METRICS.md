# Evaluation Metrics in Machine Learning

## Table of Contents
1. Why Metrics Matter
2. Classification Metrics
3. Regression Metrics
4. Metric Selection by Use-Case
5. Interview Insights

---

## 1) Why Metrics Matter

- Align model optimization with business goals
- Different metrics expose different failure modes
- Avoid misleading metrics (e.g., accuracy on imbalanced data)

---

## 2) Classification Metrics

Confusion Matrix:
```
                 Predicted
             Negative    Positive
Actual  Neg     TN         FP
        Pos     FN         TP
```

- Accuracy = (TP+TN)/(TP+TN+FP+FN): misleading with imbalance
- Precision = TP/(TP+FP): of predicted positives, fraction correct
- Recall (TPR) = TP/(TP+FN): of actual positives, fraction found
- F1 = 2·(Precision·Recall)/(Precision+Recall): balance of precision & recall
- Specificity (TNR) = TN/(TN+FP): true negative rate
- ROC-AUC: probability a random positive ranks above a random negative; insensitive to threshold
- PR-AUC: area under precision-recall curve; better for rare positives
- Log Loss / Cross-Entropy: penalizes wrong confident probabilities (calibration focus)
- Cohen’s Kappa / Matthews Correlation: agreement measures robust to imbalance

Thresholding:
- Select threshold based on business costs (FP vs FN)
- Calibrate probabilities if required (Platt, isotonic)

Class Imbalance:
- Prefer PR-AUC, F1, or cost-sensitive metrics
- Use class-weighting or threshold tuning rather than accuracy

---

## 3) Regression Metrics

- MSE: average squared error; penalizes large errors more
- RMSE: sqrt(MSE); same units as target; interpretable spread
- MAE: average absolute error; robust to outliers
- MAPE: relative error (%); undefined near zero; beware skew
- R² (coefficient of determination): proportion of variance explained; negative if worse than baseline
- MedAE / Quantile loss: robust under skewed distributions

Choice guidance:
- Outliers present → MAE/MedAE
- Penalize large misses → RMSE/MSE
- Relative error important → MAPE (ensure >0 targets)

---

## 4) Metric Selection by Use-Case

Fraud/Churn (rare events): PR-AUC, Recall@K, F1; threshold by cost
Search/Retrieval: NDCG, MRR, Recall@K, Precision@K
Medical diagnosis: High recall (sensitivity) with acceptable precision; ROC-AUC and PR-AUC
Recommendation: Hit-rate, MAP@K, NDCG@K, coverage, diversity
Forecasting: MAPE/SMAPE with caution; pinball loss (quantile) for uncertainty

Business alignment:
- Define costs of FP and FN; choose threshold to minimize expected cost
- Track leading indicators and operational constraints (latency, throughput)

---

## 5) Interview Insights

- Metric ≠ goal; pick metrics reflecting business trade-offs
- Report multiple metrics to reveal trade-offs (e.g., PR-AUC + calibration)
- Handle imbalance explicitly; accuracy can be deceptive
- For ranking problems, ROC-AUC may be acceptable while PR-AUC reveals poor performance on positives
