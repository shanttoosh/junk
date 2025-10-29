# Linear Regression

## Table of Contents
1. What is Linear Regression?
2. Assumptions and When They Hold
3. Objective and Intuition (No Code)
4. Variants and Regularization
5. Diagnostics and Pitfalls
6. Interview Insights

---

## 1) What is Linear Regression?

Linear Regression models the relationship between a continuous target variable and one or more input features as a linear combination. It estimates coefficients that best explain how the expected value of the target changes with the features.

- Simple Linear Regression: one feature → a line in 2D
- Multiple Linear Regression: many features → a hyperplane in nD

High-level goal: find the line/hyperplane that minimizes overall prediction error.

---

## 2) Assumptions and When They Hold

Linear regression relies on these standard assumptions (for valid inference and reliable generalization):

- Linearity: The expected target is a linear function of features. Use residual plots to check; if curves appear, add interactions or transformations.
- Independence: Observations do not influence each other. Time series and clustered data often violate this; use appropriate models instead (e.g., ARIMA, mixed-effects).
- Homoscedasticity: Constant variance of residuals across fitted values. Funnel-shaped residuals indicate heteroscedasticity; consider transformations or robust regression.
- Normality of errors: Residuals are approximately normal. This primarily matters for confidence intervals and hypothesis tests; with large data, CLT helps.
- No multicollinearity: Features should not be highly collinear. Use VIF/correlation matrix; mitigate with feature selection or dimensionality reduction (e.g., PCA) or regularization.

When appropriate:
- Baselines, explainability needs, or when relationships are close to linear.

When not appropriate:
- Strongly non-linear relationships without transformations; heavy interactions; complex boundaries.

---

## 3) Objective and Intuition (No Code)

Objective: minimize average squared error between predictions and targets (least squares). Geometrically, find the hyperplane that minimizes squared vertical distances to points.

Why squared error?
- Penalizes large mistakes more than small ones.
- Leads to a unique, closed-form optimum if features are full rank.

Effect of outliers: Squared errors amplify outliers; consider robust alternatives (Huber, quantile regression) if heavy-tailed noise exists.

Feature scaling: Not required for plain OLS, but crucial when adding regularization (Ridge/Lasso) and beneficial for numerical stability.

Interpretation of coefficients:
- Holding other features constant, a coefficient reflects the expected change in the target for a one-unit increase in that feature.
- Intercept: expected target when all features are zero (may be non-meaningful depending on feature scaling/centering).

---

## 4) Variants and Regularization

- Ridge (L2): Shrinks coefficients toward zero to reduce variance and improve generalization when features are correlated; never forces exact zeros.
- Lasso (L1): Performs feature selection by driving some coefficients exactly to zero; useful for high-dimensional sparse problems.
- Elastic Net (L1 + L2): Balances Lasso and Ridge advantages; helpful when correlated groups of features exist.
- Polynomial Regression: Adds non-linear terms (e.g., x^2, x·y) while keeping a linear model in parameters.

Choosing among them:
- Many correlated features → Ridge/Elastic Net
- Need feature selection → Lasso/Elastic Net
- Mild overfitting → Small L2 often stabilizes

---

## 5) Diagnostics and Pitfalls

Diagnostics (conceptual checks):
- Residual plots vs fitted values: look for randomness (no pattern) and constant spread.
- Q–Q plot of residuals: assess normality assumption for inference.
- Influence measures: identify high-leverage outliers that drive fit disproportionately.
- Multicollinearity checks: correlation matrix, VIF.

Common pitfalls:
- Extrapolation: Linear trends outside observed range are unreliable.
- Omitted variable bias: Missing key drivers can bias coefficients.
- Leakage: Using information not available at prediction time inflates metrics.
- Heteroscedastic noise: Consider transformations (log target) or weighted least squares.

---

## 6) Interview Insights

- When to prefer linear regression? Strong need for interpretability, quick baseline, and approximately linear relationships.
- How to handle non-linearity? Add interactions, polynomial terms, or move to non-linear models.
- Why regularize? To combat overfitting and multicollinearity; improves generalization.
- Business framing: Coefficients quantify impact per feature unit—useful for pricing, marketing mix modeling, and sensitivity analyses.
