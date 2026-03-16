# CatBoost (Categorical Boosting)

CatBoost is a gradient boosting framework developed by Yandex, published at NeurIPS 2018. It was specifically designed to handle **categorical features natively** without manual preprocessing, and introduces **ordered boosting** to reduce the prediction shift (overfitting) that affects all traditional gradient boosting implementations.

---

## How CatBoost Works

### The Problem CatBoost Solves — Target Leakage

In traditional gradient boosting, when computing the target statistics for a categorical feature (e.g., average price per city), the target value of the current example is used in its own encoding. This causes **target leakage** — the model sees information from the future during training.

```
Example: Encoding "city" using mean target

City     | Target | Mean encoding (WRONG — includes itself)
---------|--------|------------------------------------------
NYC      | 500    | mean(500, 300, 700) = 500  ← sees own target
NYC      | 300    | mean(500, 300, 700) = 500  ← sees own target
NYC      | 700    | mean(500, 300, 700) = 500  ← sees own target
```

### Ordered Target Statistics

CatBoost solves this by computing target statistics using only **examples that come before** the current one in a random permutation:

```
Random permutation: [example₃, example₁, example₂]

Encoding example₃: no prior NYC data → use global prior
Encoding example₁: only example₃'s target → mean(700) = 700
Encoding example₂: example₃ and example₁'s targets → mean(700, 500) = 600
```

The formula for ordered target statistics:

```
x̂ᵢ = (Σⱼ<ᵢ [xⱼ ∈ same_category] · yⱼ + a · p) / (Σⱼ<ᵢ [xⱼ ∈ same_category] + a)
```

where:
- The sum is only over examples `j` that appear **before** example `i` in the permutation
- `a` = smoothing parameter (prevents unstable estimates for rare categories)
- `p` = global prior (average target across all training data)

### Ordered Boosting

CatBoost extends the ordering principle to the entire boosting process. In standard gradient boosting, the residuals for example `i` are computed using a model trained on all data **including example `i`** — another form of target leakage.

CatBoost's ordered boosting:

1. Generate a random permutation `σ` of the training data
2. For each example `i`, compute its residual using a model trained only on examples `σ(1), ..., σ(i-1)`
3. This means each example's gradient is computed from a model that **never saw that example**

This is computationally expensive (requires maintaining `n` different models), so CatBoost approximates it by maintaining `log₂(n)` models trained on exponentially growing subsets.

### Oblivious Decision Trees

CatBoost uses **oblivious (symmetric) decision trees** as its base learner. In an oblivious tree, the same splitting condition is used across all nodes at the same depth:

```
Standard tree:                 Oblivious tree:
     [age > 30]                    [age > 30]
    /          \                  /          \
[income>50K] [city=NYC]     [income>50K]  [income>50K]  ← same split at each level
```

**Advantages of oblivious trees:**
- **Fast inference** — the tree is essentially a lookup table (2^depth entries)
- **Less overfitting** — fewer unique tree structures possible
- **GPU-friendly** — uniform structure enables efficient parallel computation
- **Built-in regularization** — the symmetry constraint limits model complexity

### Handling Categorical Features

CatBoost supports three approaches for categorical features:

1. **Ordered target statistics** (default) — described above, no preprocessing needed
2. **One-hot encoding** — for low-cardinality features (≤ `one_hot_max_size` unique values)
3. **Combinations** — CatBoost automatically tries combinations of categorical features (e.g., city × gender) to capture interactions

---

## When to Use CatBoost

**Best suited for:**
- Datasets with many categorical features (country, product type, user segment, etc.)
- Situations where you want minimal preprocessing — no need for label encoding or one-hot encoding
- Small to medium datasets where overfitting is a concern (ordered boosting helps)
- When you want strong defaults — CatBoost requires less hyperparameter tuning
- Datasets with categories that have natural ordering (CatBoost can learn this)

**Not ideal for:**
- Pure numerical datasets with no categorical features — XGBoost or LightGBM may be simpler
- Very large datasets (>10M rows) — CatBoost's ordered boosting overhead slows training
- When you need the absolute fastest training time — LightGBM is faster
- Production systems with strict latency requirements (oblivious trees are fast for inference, but CatBoost's preprocessing adds overhead)

**Data types that work well:**
- E-commerce (product categories, user demographics, location)
- Banking and finance (account type, country, occupation)
- Healthcare (diagnosis codes, treatment categories, demographics)
- Marketing (campaign type, channel, customer segment)
- Any tabular data where you'd normally spend significant time encoding categoricals

**Key hyperparameters for CatBoost specifically:**

| Parameter | Default | Typical Range | Why It Matters |
|-----------|---------|---------------|----------------|
| `iterations` | 1000 | 100–5000 | Number of trees (use with early stopping) |
| `depth` | 6 | 4–10 | Depth of oblivious trees |
| `learning_rate` | Auto | 0.01–0.3 | CatBoost can auto-tune this |
| `l2_leaf_reg` | 3 | 1–10 | L2 regularization on leaf values |
| `border_count` | 254 | 32–255 | Number of bins for numerical features |
| `one_hot_max_size` | 2 | 2–25 | Max categories for one-hot (rest use target stats) |
| `od_type` | IncToDec | — | Overfitting detector type |
| `od_wait` | 20 | 20–100 | Patience for early stopping |

---

## Experiments

### Regression — Melbourne Housing Prices (`reg/catboost_reg.py`)

**Dataset:** Kaggle Melbourne housing dataset (`melb_data.csv`) — 13,580 properties, 21 features

**Features include:** Suburb, Address, Rooms, Type, Price, Method, SellerG, Distance, Postcode, Bedroom, Bathroom, Car, Landsize, BuildingArea, YearBuilt, CouncilArea, Latitude, Longitude, Regionname

**Results:**

| Metric | Value |
|--------|-------|
| RMSE | **236,505** |
| R² Score | **0.860** |
| Iterations | 2,000 (best at iteration 1,999) |

**Training progression:**

| Iteration | Train Loss | Test Loss | Status |
|-----------|-----------|-----------|--------|
| 0 | 622,346 | 610,859 | Starting |
| 200 | 245,310 | 264,271 | Rapid improvement |
| 600 | 190,283 | 246,027 | Slowing down |
| 1,000 | 157,969 | 240,392 | Test loss plateauing |
| 1,400 | 136,099 | 238,473 | Minimal test improvement |
| 1,999 | 112,501 | 236,505 | Best test loss |

**Interpretation:** The model explains 86% of Melbourne housing price variance — the highest R² across all experiments in this repository. The training loss continued to decrease (622K → 112K) while test loss plateaued after ~600 iterations (264K → 236K), showing the classic overfitting pattern. Despite this, the model kept improving on test data through all 2,000 iterations, suggesting the regularization from oblivious trees was effective.

CatBoost excels here because the Melbourne dataset is rich in categorical features — Suburb (hundreds of unique values), CouncilArea, Type, Method, SellerG, Regionname. Traditional approaches would require extensive encoding, but CatBoost's ordered target statistics handle these natively and capture location-price interactions that numerical-only models miss.

### Classification — Bank Churn Prediction (`cla/cust_churn.py`)

**Dataset:** Bank Customer Churn Prediction dataset — 10,000 customers, 12 features, binary classification

**Features:** customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, churn

**Data quality:** Zero missing values across all 12 columns.

**Two approaches compared:**

#### Approach 1: Direct CatBoost (DataFrame + categorical feature names)

```python
cat_features = ['country', 'gender']  # Passed by name
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Churn (0) | 0.88 | 0.97 | 0.92 | 1,593 |
| Churn (1) | **0.82** | 0.49 | **0.61** | 407 |
| **Accuracy** | | | **87%** | 2,000 |

#### Approach 2: Pipeline CatBoost (categorical feature indices)

```python
cat_features = [0, 1]  # Passed by index after preprocessing
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Churn (0) | 0.88 | 0.96 | 0.92 | 1,593 |
| Churn (1) | 0.74 | 0.49 | 0.59 | 407 |
| **Accuracy** | | | **86%** | 2,000 |

**Training details (Direct approach):**
- Early stopping triggered at **iteration 86** (50-iteration patience)
- Best test accuracy: 0.874
- Overfitting detector stopped training — model would have degraded with more iterations

**Why Direct CatBoost won:**
The key difference is churn precision — **0.82 vs 0.74**. When passing categorical features by name directly to CatBoost, the ordered target statistics encoding preserves the natural structure of categories (e.g., "Germany" vs "France" vs "Spain" carry geographic and economic meaning). The pipeline approach with index-based identification may lose some of this metadata during preprocessing.

**Comparison with LightGBM (same dataset):**

| Metric | CatBoost | LightGBM | Winner |
|--------|----------|----------|--------|
| Accuracy | **87%** | 86% | CatBoost |
| Churn Precision | **0.82** | 0.74 | CatBoost (+10.8%) |
| Churn Recall | 0.49 | 0.48 | Tied |
| Churn F1 | **0.61** | 0.58 | CatBoost (+5.2%) |
| Iterations | **87** (early stop) | 100 (default) | CatBoost (fewer) |

CatBoost achieved better results with fewer iterations thanks to ordered boosting and early stopping. The precision gain on churn class (0.82 vs 0.74) means CatBoost makes fewer false churn predictions, which is valuable in practice — incorrectly flagging loyal customers as churners leads to unnecessary retention spending.

---

## Usage

```bash
# Regression — Melbourne housing (~20 seconds)
cd reg && python3 catboost_reg.py

# Classification — Bank churn (~2 seconds with early stopping)
cd ../cla && python3 cust_churn.py
```

**Note:** CatBoost generates a `catboost_info/` folder during training containing training logs, learning curves, and metadata. This is auto-generated and can be added to `.gitignore`.

---

## References

- Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* Advances in Neural Information Processing Systems (NeurIPS).
- Dorogush, A.V. et al. (2018). *CatBoost: Gradient Boosting with Categorical Features Support.* Workshop on ML Systems at NeurIPS.
- [CatBoost Documentation](https://catboost.ai/docs/)
