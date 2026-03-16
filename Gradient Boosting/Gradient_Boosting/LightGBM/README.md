# LightGBM (Light Gradient Boosting Machine)

LightGBM is a gradient boosting framework developed by Microsoft Research. It was designed to be significantly faster than XGBoost while maintaining (or improving) accuracy, particularly on large-scale datasets. Published at NeurIPS 2017, it introduced two key innovations: **Gradient-based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)**.

---

## How LightGBM Works

### Tree Building — Leaf-Wise Growth

LightGBM's most distinctive feature is **leaf-wise** (best-first) tree growth. Instead of growing all nodes at the same level, it always splits the leaf with the highest loss reduction (gain), regardless of depth.

```
Leaf-wise growth (LightGBM):

        [Root]
       /      \
    [A]        [B]          ← B has higher gain, so split B first
              /    \
           [B1]    [B2]     ← B2 has highest gain now, split B2
                  /    \
               [B2a]  [B2b] ← Deeper where it matters
```

**Pros:** Converges faster — reaches the same loss in fewer iterations because it always optimizes the highest-gain split first.

**Cons:** Can overfit on small datasets because trees can grow very deep in one region. Mitigate with `max_depth` or `num_leaves` constraints.

### Gradient-based One-Side Sampling (GOSS)

Traditional gradient boosting uses all data points for every split. GOSS keeps all instances with **large gradients** (poorly predicted, most informative) and randomly samples from instances with **small gradients** (well predicted, less informative).

```
Algorithm:
1. Sort instances by absolute gradient value
2. Keep top a% instances (large gradients — high information)
3. Randomly sample b% from remaining instances (small gradients)
4. Amplify the sampled instances by (1-a)/b to maintain distribution
```

With `a = 20%` and `b = 10%`, LightGBM trains on only 30% of data per iteration while preserving the gradient distribution. This gives a 3× speedup with minimal accuracy loss.

### Exclusive Feature Bundling (EFB)

Many real-world datasets have sparse features that are rarely non-zero simultaneously (e.g., one-hot encoded features). EFB bundles these **mutually exclusive features** into single features, reducing the effective number of features.

```
Before EFB:  [is_male, is_female, is_NYC, is_LA, is_Chicago]  → 5 features
After EFB:   [gender_bundle, city_bundle]                       → 2 features
```

This reduces split-finding complexity from `O(#features)` to `O(#bundles)`.

### Histogram-Based Split Finding

Instead of sorting feature values for every split (O(n·log(n))), LightGBM **bins continuous values into discrete histograms** (O(n) to build, O(#bins) to find best split).

```
Raw values:     [1.2, 3.7, 2.1, 5.5, 4.3, 1.8, 3.2, 4.9]
Binned (4 bins): [ 0,   2,   1,   3,   2,   0,   2,   3 ]
```

Default is 255 bins — enough for most distributions while being 10× faster than exact split finding.

---

## When to Use LightGBM

**Best suited for:**
- Large datasets (100K+ rows) where training speed matters
- Datasets with many features (EFB automatically handles sparse features)
- High-cardinality categorical features (native encoding avoids one-hot explosion)
- Production systems requiring frequent retraining (fastest gradient boosting framework)
- When you need a good model quickly with less tuning

**Not ideal for:**
- Very small datasets (<1K rows) — leaf-wise growth can overfit aggressively
- When you need reproducible results across platforms (histogram binning can produce slightly different results)
- Heavily categorical data with no numerical features — CatBoost may be better

**Data types that work well:**
- Click-through rate prediction (large-scale, many features)
- Financial time series (high-frequency, needs fast retraining)
- NLP feature matrices (sparse TF-IDF or embedding features)
- Recommendation systems (large user-item feature matrices)

**Key hyperparameters for LightGBM specifically:**

| Parameter | Default | Typical Range | Why It Matters |
|-----------|---------|---------------|----------------|
| `num_leaves` | 31 | 20–100 | Primary control for model complexity (not max_depth) |
| `min_data_in_leaf` | 20 | 10–100 | Prevents overfitting from leaf-wise growth |
| `feature_fraction` | 1.0 | 0.6–0.9 | Column subsampling per tree |
| `bagging_fraction` | 1.0 | 0.6–0.9 | Row subsampling per tree |
| `lambda_l1` | 0 | 0–10 | L1 regularization |
| `lambda_l2` | 0 | 0–10 | L2 regularization |

The key insight: **control complexity with `num_leaves`, not `max_depth`**. A tree with `max_depth=7` has up to 128 leaves, but `num_leaves=31` allows deep trees that are still constrained in total complexity.

---

## Experiments

### Regression — Laptop Price Prediction (`reg/laptop_lgbm.py`)

**Dataset:** Kaggle laptop price dataset (`laptop_price.csv`) — 1,303 laptops, 12 features

**Results:**

| Metric | LightGBM | XGBoost (same data) | Improvement |
|--------|----------|---------------------|-------------|
| MAE | **176.55** | 212.78 | -17.0% |
| MSE | **103,145.13** | 138,072.54 | -25.3% |
| RMSE | **321.16** | 371.58 | -13.6% |
| R² Score | **0.800** | 0.728 | +9.9% |

**Training details:**
- 1,042 training samples, 12 features used
- Auto-selected column-wise multi-threading
- Total bins: 866

**Why LightGBM won here:** The laptop dataset has features with varying importance levels — Ram and TypeName dominate while others contribute marginally. LightGBM's leaf-wise growth naturally focuses more splits on the high-information regions (where Ram and TypeName interact to determine price), while XGBoost's level-wise approach allocates equal splitting effort across all regions. On a small dataset like this (1,303 rows), the efficiency of leaf-wise growth in finding the right splits outweighs the overfitting risk.

### Classification — Bank Churn Prediction (`cla/churn_lgbm.py`)

**Dataset:** Bank Customer Churn Prediction dataset — 10,000 customers, 12 features, binary classification

**Class distribution:** 20.4% churn (1,630) vs 79.6% no churn (6,370) — imbalanced

**Results:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Churn (0) | 0.88 | 0.96 | 0.92 | 1,593 |
| Churn (1) | 0.74 | 0.48 | 0.58 | 407 |
| **Overall Accuracy** | | | **86%** | **2,000** |

**Training details:**
- 8,000 training samples, 13 features used
- Binary classification with BoostFromScore: `pavg=0.2038 → initscore=-1.363`
- Total bins: 865

**Interpretation:** The initial score of -1.363 (log-odds of the average churn rate) shows LightGBM correctly initializes from the class prior. The model is strong at identifying non-churners (96% recall) but conservative on churn prediction (48% recall). This is the classic precision-recall tradeoff in imbalanced datasets — the model avoids false churn predictions at the cost of missing real churners. To improve churn recall, you could: adjust the classification threshold below 0.5, use `scale_pos_weight` to penalize missed churners, or apply SMOTE oversampling.

---

## Usage

```bash
cd reg && python3 laptop_lgbm.py     # Regression metrics
cd ../cla && python3 churn_lgbm.py   # Classification report + confusion matrix
```

---

## References

- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems (NeurIPS).
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
