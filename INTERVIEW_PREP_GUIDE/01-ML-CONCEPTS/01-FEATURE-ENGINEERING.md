# Feature Engineering

## Table of Contents
1. [What is Feature Engineering?](#what-is-feature-engineering)
2. [Feature Selection](#feature-selection)
3. [Feature Transformation](#feature-transformation)
4. [Feature Creation](#feature-creation)
5. [Common Techniques](#common-techniques)
6. [Interview Insights](#interview-insights)

---

## What is Feature Engineering?

### Definition

**Feature Engineering** is the process of transforming raw data into features that better represent the underlying problem to predictive models, resulting in improved model accuracy on unseen data.

**Key Principle**: The quality and quantity of your features determine the upper limit of what your model can learn. Better features → Better model performance.

### Why It Matters

**Impact on Model Performance**:
- Feature engineering can improve model accuracy by 30-40%
- Poor features require complex models to achieve decent performance
- Good features allow simple models to excel

**Real-World Analogy**: 
- Poor features = Asking a student to solve calculus problems in a foreign language
- Good features = Teaching the same student calculus in their native language

The data is the same, but the representation makes all the difference.

---

## Feature Selection

### What is Feature Selection?

**Goal**: Identify and keep only the most relevant features for modeling.

**Why Select Features?**:
1. **Reduce Overfitting**: Fewer features → less complex model → better generalization
2. **Faster Training**: Less data to process
3. **Lower Cost**: In cloud, fewer features means less storage and compute
4. **Interpretability**: Easier to understand model decisions
5. **Curse of Dimensionality**: High-dimensional spaces are sparse and misleading

### Selection Methods

#### 1. Filter Methods

**Approach**: Use statistical measures to score features, independent of model.

**Techniques**:
- **Correlation**: Remove features highly correlated with others
- **Chi-square test**: For categorical features (independence test)
- **ANOVA F-test**: For numerical features (variance analysis)
- **Mutual Information**: Measures dependency between features

**Pros**: Fast, computationally cheap, model-agnostic  
**Cons**: Doesn't consider feature interactions

**Example (Conceptual)**:
```
Before: 50 features, many correlated
After: 25 features, each provides unique information
Result: Simpler model, better generalization
```

#### 2. Wrapper Methods

**Approach**: Use model performance to select features.

**Techniques**:
- **Forward Selection**: Start with no features, add best one at a time
- **Backward Elimination**: Start with all features, remove worst one at a time
- **Recursive Feature Elimination (RFE)**: Train model, remove least important, repeat

**Process (Forward Selection)**:
```
Step 1: Try each feature individually → Best: "tenure"
Step 2: Try "tenure" + each other feature → Best: "tenure + monthly_charges"
Step 3: Try "tenure + monthly_charges" + each remaining → Continue...
Stop when adding features doesn't improve performance
```

**Pros**: Considers feature interactions, uses actual performance  
**Cons**: Computationally expensive, prone to overfitting

#### 3. Embedded Methods

**Approach**: Selection is built into the learning algorithm.

**Example - Lasso Regularization**:
- L1 regularization automatically drives some coefficients to zero
- Non-zero coefficients = selected features
- No separate selection step needed

**Example - Tree-based Methods**:
- Decision trees naturally rank feature importance
- Use feature importance scores to select top K features
- Random Forest aggregates importances across trees

**Pros**: Automatic, computationally efficient  
**Cons**: Tied to specific algorithms

---

## Feature Transformation

### Why Transform Features?

**Problems**:
- Different scales (age: 20-80, income: 30000-200000)
- Skewed distributions
- Outliers affecting model

**Solutions**: Transformations normalize and stabilize the data.

### Scaling Techniques

#### 1. Standardization (Z-score Normalization)

**Formula**: $z = \frac{x - \mu}{\sigma}$

**Effect**: Transforms data to have mean=0, std=1

**When to use**:
- Algorithms sensitive to scale (SVM, neural networks)
- Features measured on different units

**Intuition**: 
```
Before: Age [20, 80], Income [30000, 200000]
After: Both are on same scale [-2, 2]
```

#### 2. Min-Max Normalization

**Formula**: $x' = \frac{x - \min}{\max - \min}$

**Effect**: Scales data to [0, 1] range

**When to use**:
- Neural networks (activations expect [0,1])
- Preserving relationships for distances

**Intuition**: Everything compressed to 0-1 range

#### 3. Robust Scaling

**Formula**: Uses median and IQR instead of mean and std

**When to use**: When you have outliers

**Why**: Outliers don't affect median/IQR as much as mean/std

### Categorical Encoding

#### 1. One-Hot Encoding

**How it works**: Create binary columns for each category

```
Original:
Gender: [Male, Female, Male, Female]

Encoded:
Gender_Male: [1, 0, 1, 0]
Gender_Female: [0, 1, 0, 1]
```

**When to use**: 
- Nominal categories (no order)
- High-cardinality okay (<50 categories)

**Pros**: Preserves information, no ordering imposed  
**Cons**: Increases dimensionality (curse of dimensionality)

#### 2. Label Encoding

**How it works**: Assign integer labels

```
Original: [Red, Blue, Green]
Encoded: [0, 1, 2]
```

**When to use**: 
- Ordinal categories (size: S, M, L)
- Tree-based models (handle integers well)

**Pros**: Doesn't increase dimensionality  
**Cons**: Imposes artificial order on nominal categories

#### 3. Target Encoding (Mean Encoding)

**How it works**: Replace category with mean of target variable

```
Category | Target
A        | 1
A        | 0
B        | 1
B        | 1
B        | 0

Target Encoded:
A → 0.5 (mean of [1, 0])
B → 0.67 (mean of [1, 1, 0])
```

**When to use**: High-cardinality categories (city names, product IDs)

**Pros**: Captures predictive power  
**Cons**: Can cause overfitting if not done carefully

---

## Feature Creation

### Polynomial Features

**Idea**: Create new features from combinations of existing ones

**Example**:
```
Original: x1, x2
Polynomial (degree=2): x1, x2, x1², x1·x2, x2²
```

**When to use**: 
- Non-linear relationships suspected
- Domain knowledge suggests interactions

**Intuition**: 
- If feature A is important and B is important, their interaction might be too
- Example: Income × Age (earning potential changes with age)

### Binning (Discretization)

**Idea**: Convert continuous features to categorical

```
Age: 20, 25, 30, 45, 55, 70
  ↓
Bins: [18-30, 31-50, 51-65, 65+]
  ↓
AgeGroup: Young, Middle, Senior, Elderly
```

**Why**: 
- Non-linear relationships (age vs risk)
- Robust to outliers
- Easier interpretation

### Aggregation Features

**Idea**: Create features from groups

**Examples**:
- Customer: Count of orders, average order value
- Time series: Rolling averages, differences
- Group by: Mean/max/min per category

**Example (Churn)**:
```
Per customer:
- Total contracts signed
- Average months per contract
- Total revenue
```

---

## Common Techniques

### Handling Missing Values

**Strategies**:

1. **Deletion**: Remove rows/columns with many missing values
2. **Mean/Median/Mode**: Fill with central tendency (simple but sometimes naive)
3. **Forward/Backward Fill**: For time series
4. **Predictive Imputation**: Use model to predict missing values
5. **Special Indicator**: Create "missing" category

**Decision factors**:
- Amount of missingness (<5% vs >50%)
- Type of missingness (random vs systematic)
- Domain knowledge (why missing?)

### Handling Outliers

**Detection**:
- Statistical: Z-score > 3 or IQR method
- Domain knowledge: Impossible values

**Treatment**:
- **Capping**: Limit extreme values
- **Transformation**: Log, sqrt to reduce impact
- **Removal**: Only if clearly errors
- **Binning**: Convert to categories

### Feature Interaction

**Why**: Individual features might not be predictive, but combinations are

**Examples**:
- Total revenue vs Monthly charges (both alone vs together)
- Gender × Age (demographic interactions)
- Contract type × Tenure (loyalty interactions)

**How to discover**: 
- Domain expertise
- Correlation heatmaps
- Feature importance from models

---

## Interview Insights

### Common Questions

**Q1: What's the difference between feature selection and feature engineering?**

**Answer**: 
- **Selection**: Choosing which features to keep (removing bad ones)
- **Engineering**: Creating new features or transforming existing ones (adding good ones)

Analogy: 
- Selection = Choosing best ingredients from your pantry
- Engineering = Combining ingredients to create new dishes

**Q2: How do you decide between standardization and normalization?**

**Answer**:
- **Standardization (Z-score)**: When features have different scales, algorithms sensitive to distance (SVM, neural networks). Assumes normal distribution.
- **Min-Max**: When you need [0,1] range, NN activations. Preserves original distribution shape.

**General rule**: Use standardization unless you have a specific reason for [0,1] range.

**Q3: When would you use target encoding vs one-hot encoding?**

**Answer**:

| Aspect | One-Hot | Target Encoding |
|--------|---------|----------------|
| **Use when** | Low cardinality | High cardinality |
| **Pros** | Simple, preserves all info | Captures predictive signal |
| **Cons** | Increases dimensions | Risk of overfitting |
| **Best for** | <50 categories | 100+ categories |

Target encoding is powerful for high-cardinality (city, product ID) but requires care to avoid leakage.

**Q4: Explain the impact of feature engineering on model complexity.**

**Answer**:
- **Good features**: Simple models (linear) can work well
- **Poor features**: Require complex models (deep neural networks) just to learn patterns

Example:
- Without feature engineering: Complex NN needed, still struggles
- With feature engineering: Logistic regression excels

**Business value**: Simpler models = faster, cheaper, more interpretable

### Common Pitfalls

❌ **Feature leakage**: Including information that won't be available at prediction time  
✅ **Solution**: Only use features available in production

❌ **Over-engineering**: Creating too many features without understanding impact  
✅ **Solution**: Start simple, add features based on performance

❌ **Ignoring categorical variables**: Dropping them or using ordinal encoding incorrectly  
✅ **Solution**: Proper encoding strategy based on cardinality and type

---

**Next**: [Dimensionality Reduction →](02-DIMENSIONALITY-REDUCTION.md)

