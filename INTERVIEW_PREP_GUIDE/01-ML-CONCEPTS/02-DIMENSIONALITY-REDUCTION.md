# Dimensionality Reduction

## Table of Contents
1. [What is Dimensionality Reduction?](#what-is-dimensionality-reduction)
2. [Curse of Dimensionality](#curse-of-dimensionality)
3. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
4. [Alternative Techniques](#alternative-techniques)
5. [When to Use](#when-to-use)
6. [Interview Insights](#interview-insights)

---

## What is Dimensionality Reduction?

### Definition

**Dimensionality Reduction** is the process of reducing the number of features while preserving most of the important information.

**Key Idea**: Fewer dimensions but most information retained.

### Why Reduce Dimensions?

**Benefits**:
1. **Curse of Dimensionality**: Performance degrades in high dimensions
2. **Visualization**: Can only plot 2-3 dimensions
3. **Storage**: Less memory required
4. **Computation**: Faster algorithms
5. **Noise Reduction**: Removes redundant information
6. **Overfitting Prevention**: Simpler models generalize better

**Real-World Analogy**: 
- Original: 1000 features describing a house
- Reduced: 10 features capturing essential information
- Result: Easier to understand, faster to process, better predictions

---

## Curse of Dimensionality

### The Problem

**As dimensions increase**:
- Points become equidistant (everyone is the same distance from everyone)
- Volume concentrates at edges
- Nearest neighbor algorithm loses meaning
- Need exponentially more data

**Intuitive Example**: 
- In 2D: Can have 100 data points in a 10×10 grid, density is reasonable
- In 100D: Need 10^100 points to maintain same density!

**Impact on ML**:
- Nearest neighbor: "Near" loses meaning in high dimensions
- Distance metrics: Euclidean distance becomes less discriminative
- Statistical significance: Need way more samples
- Overfitting: Models memorize noise

---

## Principal Component Analysis (PCA)

### Core Concept

**PCA** finds the directions (principal components) where data varies the most.

**Geometric Intuition**:
```
Imagine data points scattered in 3D space
PCA finds: 
- PC1: Direction with most variance
- PC2: Direction with second most variance (orthogonal to PC1)
- PC3: Direction with least variance
```

### How PCA Works

**Step-by-Step Process**:

**1. Center the Data**
```
Original: x ∈ ℝⁿ
Centered: x' = x - μ (mean)
```

**2. Compute Covariance Matrix**
```
C = (1/m) X^T X
Where X is centered data
```

Shows how features vary together.

**3. Find Eigenvectors and Eigenvalues**
```
C V = λ V
```

Eigenvectors = principal components  
Eigenvalues = variance captured by each component

**4. Select Top-K Components**
```
Choose K such that captures 95% variance

Example:
PC1: 60% variance
PC2: 25% variance  
PC3: 10% variance
Total: 95% → Use PC1, PC2, PC3
```

**5. Project Data**
```
Y = X V_k
Where V_k are top-K eigenvectors
```

### Mathematical Formulation

**Objective Function**:
Maximize variance of projected data

$$\max_{\mathbf{w}} \text{Var}(\mathbf{w}^T \mathbf{x})$$

Subject to: $||\mathbf{w}|| = 1$

**Solution**: Eigenvectors of covariance matrix

### Example (Conceptual)

**Original Data**: 1000 features describing customers

**PCA Transformation**:
- PC1 captures 40% variance → "Overall customer value"
- PC2 captures 30% variance → "Engagement level"
- PC3 captures 20% variance → "Payment behavior"
- Total: 90% variance in 3 dimensions

**Result**: 
- From 1000 dimensions → 3 dimensions
- Lost only 10% information
- Much easier to visualize and model

### Key Properties

**1. Uncorrelated Components**
Principal components are orthogonal (independent)

**2. Maximizes Variance**
Each PC captures maximum possible variance

**3. Linear Transformation**
PCA is a linear transformation (faster, but limited)

**4. Non-Interpretable**
Components are linear combinations, not original features

---

## Alternative Techniques

### 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose**: Non-linear dimensionality reduction for visualization

**How it works**: 
- Preserves local neighborhood structure
- Useful for visualization (2D plots)

**Limitations**:
- Expensive computation
- Random initialization (results vary)
- Not for general dimensionality reduction (only visualization)

**When to use**: Visualizing high-dimensional data (e.g., word embeddings, clusters)

### 2. Feature Selection

**Approach**: Keep original features, just remove less important ones

**Methods**:
- **Correlation**: Remove highly correlated features
- **Variance**: Remove low-variance features
- **Univariate tests**: Statistical tests per feature
- **Model-based**: Use feature importance from models

**Pros**: 
- Maintains interpretability (using original features)
- No transformation required

**Cons**: 
- Might discard potentially useful features
- No feature combination benefits

### 3. Autoencoders

**Approach**: Neural network that learns to compress and reconstruct

**Architecture**:
```
Input (1000D) 
  ↓
Encoder (compresses to 50D)
  ↓
Decoder (reconstructs to 1000D)
```

**Training**: Minimize reconstruction error

**Pros**:
- Non-linear (can capture complex patterns)
- Learns meaningful representations

**Cons**:
- Requires lots of data
- Computationally expensive
- Less interpretable

---

## When to Use

### Use PCA When:

✅ **High dimensionality**: Hundreds/thousands of features  
✅ **Multicollinearity**: Features highly correlated  
✅ **Visualization needed**: Plot in 2D/3D  
✅ **Preprocessing for ML**: Reduce features before modeling  
✅ **Noise reduction**: Remove redundant information

### Don't Use PCA When:

❌ **Interpretability critical**: Components aren't original features  
❌ **Non-linear relationships**: PCA only handles linear  
❌ **Very small dataset**: May lose too much information  
❌ **Outliers present**: Sensitive to outliers

### Decision Framework

```
Many features? → Yes → Have correlations? → Yes → Use PCA
                                          ↓ No → Feature selection
↓ No → Keep all features
```

---

## Interview Insights

### Common Questions

**Q1: Explain PCA in simple terms.**

**Answer**: 
PCA finds the most important directions in your data. Imagine you have data scattered in 3D. You rotate your viewpoint to see the data from the most informative angle. That new angle is your first principal component. PCA does this mathematically, creating new features (principal components) that are combinations of original features and capture most of the variation in the data.

**Q2: What are the assumptions of PCA?**

**Answer**:
1. **Linearity**: Data varies linearly (not curved relationships)
2. **Large variance = important**: Assumes variance indicates importance
3. **Mean and variance sufficient**: Doesn't model higher-order relationships
4. **Linearly independent features**: Original features aren't perfectly correlated

If violated: Consider non-linear methods (Kernel PCA, t-SNE) or autoencoders.

**Q3: How do you choose number of components?**

**Answer**:
**Scree Plot Method**:
- Plot eigenvalues vs component number
- Look for "elbow" where eigenvalues plateau
- Choose components before elbow

**Variance Threshold Method**:
- Calculate cumulative variance explained
- Choose K such that >95% variance captured
- Most common in practice

**Cross-Validation Method**:
- Train models with different K
- Choose K that maximizes downstream task performance
- Most rigorous but computationally expensive

**Q4: What's the difference between PCA and feature selection?**

**Answer**:

| Aspect | PCA | Feature Selection |
|--------|-----|-------------------|
| **Output** | Transformed features (linear combinations) | Original features |
| **Interpretability** | Low (abstract components) | High (original meaning) |
| **Information Loss** | Distributed across components | All-or-nothing per feature |
| **Use Case** | Dimensionality reduction, visualization | When meaning matters |

Choose PCA for visualization, feature selection when interpretability matters.

### Common Pitfalls

❌ **Standardizing after PCA**: Must standardize BEFORE PCA  
✅ **Solution**: Always center and scale data first

❌ **Using PCA for categorical features**: Doesn't work well  
✅ **Solution**: Encode categoricals first, or use specialized methods

❌ **Applying PCA to all features blindly**: May remove important low-variance features  
✅ **Solution**: Understand your data first, some low-variance features might be important

---

**Next**: [Linear Regression →](03-LINEAR-REGRESSION.md)

