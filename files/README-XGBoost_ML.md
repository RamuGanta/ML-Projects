# XGBoost (Extreme Gradient Boosting)

XGBoost is a regularized gradient boosting framework developed by Tianqi Chen at the University of Washington. It became the dominant algorithm in Kaggle competitions from 2015–2019 and remains a production standard for tabular data.

---

## How XGBoost Works

### Tree Building — Level-Wise Growth

XGBoost builds trees **level by level** (breadth-first). At each level, all leaf nodes are split simultaneously. This produces balanced trees of a fixed depth, which provides a natural regularization effect — the tree can't grow arbitrarily deep in one region while staying shallow in another.

```
Level-wise growth (XGBoost):

        [Root]              ← Level 0: split all
       /      \
    [L1]      [L1]          ← Level 1: split all at this level
   /    \    /    \
 [L2] [L2] [L2] [L2]       ← Level 2: split all at this level
```

Compare this with LightGBM's leaf-wise approach, which always splits the leaf with the highest gain — faster convergence but higher overfitting risk.

### The XGBoost Objective Function

XGBoost's key innovation is its **regularized objective**:

```
Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
```

where:
- `L(yᵢ, ŷᵢ)` is the training loss (how well the model fits)
- `Ω(fₖ)` is the regularization term (how complex the model is)

The regularization for each tree is:

```
Ω(f) = γ · T + ½ · λ · Σ wⱼ²
```

where:
- `T` = number of leaves in the tree
- `wⱼ` = weight (prediction value) of leaf `j`
- `γ` = penalty for adding a new leaf (controls tree size)
- `λ` = L2 regularization on leaf weights (controls magnitude)

This is what makes XGBoost different from basic gradient boosting — the regularization terms explicitly penalize complexity, reducing overfitting.

### Split Finding — Gain Calculation

For each potential split, XGBoost computes the **gain**:

```
Gain = ½ · [G_L² / (H_L + λ) + G_R² / (H_R + λ) - (G_L + G_R)² / (H_L + H_R + λ)] - γ
```

where:
- `G_L, G_R` = sum of gradients in left/right child
- `H_L, H_R` = sum of hessians (second derivatives) in left/right child
- `λ` = regularization parameter
- `γ` = minimum loss reduction to make a split

If `Gain < 0`, the split is not made — this is **pruning built into the training process**.

### Handling Missing Values

XGBoost learns the **optimal default direction** for missing values during training. For each split, it tries sending missing values both left and right, and picks whichever direction reduces the loss more. This means you don't need to impute missing values before training.

---

## When to Use XGBoost

**Best suited for:**
- Medium-sized tabular datasets (1K–1M rows)
- Numerical and low-cardinality categorical features
- Problems where you need fine-grained control over regularization
- When you want the most well-documented, community-supported framework
- Production systems with established XGBoost deployment pipelines

**Not ideal for:**
- Very large datasets (>10M rows) — LightGBM is faster
- High-cardinality categorical features (hundreds of unique values) — CatBoost handles these natively
- Small datasets (<1K rows) — risk of overfitting even with regularization

**Data types that work well:**
- Structured/tabular data with numerical features
- Financial data (credit scoring, fraud detection)
- Sensor data and time series features (engineered)
- Healthcare data (patient features → outcome prediction)

---

## Experiments

### Regression

#### 1. Laptop Price Prediction (`XGBoost_reg/xgb_reg_laptop.py`)

**Dataset:** Kaggle laptop price dataset (`laptop_price.csv`) — 1,303 laptops with 12 features

**Features:** Ram, TypeName, Inches, ScreenResolution, Cpu, Memory, Gpu, OpSys, Weight, Product, Company, laptop_ID

**Results:**

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 212.78 |
| Mean Squared Error | 138,072.54 |
| Root Mean Squared Error | 371.58 |
| R² Score | **0.728** |

**Feature Importance Analysis:**

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| Ram | ~0.40 | RAM capacity is the primary price driver — directly determines what workloads a laptop can handle |
| TypeName | ~0.32 | Laptop category (Gaming, Ultrabook, Notebook) defines the price tier and target market |
| Inches | ~0.05 | Screen size contributes to price but less than specs |
| Memory | ~0.04 | Storage type/size has moderate impact |
| Gpu | ~0.04 | GPU matters mainly for gaming/creative laptops |

**Interpretation:** The model explains 72.8% of laptop price variance. The remaining ~27% likely comes from brand premium effects, specific model positioning, and market timing — factors not fully captured by hardware specs alone. The RMSE of 371.58 means predictions are off by roughly $370 on average, which is reasonable given the wide price range in the dataset.

#### 2. Uber Fare Prediction (`XGBoost_reg/xgb_reg_uber.py`)

**Dataset:** Kaggle Uber fare dataset (`uber.csv`)

**Features:** pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count, pickup_datetime, key

**Results:**

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 2.54 |
| Mean Squared Error | 32.09 |
| Root Mean Squared Error | 5.66 |
| R² Score | **0.692** |

**Feature Importance Analysis:**

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| dropoff_longitude | ~0.41 | Longitude encodes east-west distance — critical for fare calculation |
| pickup_longitude | ~0.30 | Combined with dropoff, these two features encode trip distance |
| pickup_latitude | ~0.13 | North-south component of trip distance |
| dropoff_latitude | ~0.09 | Supplements longitude for distance calculation |
| pickup_datetime | ~0.02 | Minimal — suggests base fare, not surge, dominates this dataset |

**Interpretation:** Geographic coordinates dominate because fare is fundamentally a function of distance. The model captures ~69% of fare variance. The low importance of `pickup_datetime` suggests this dataset doesn't capture strong surge pricing patterns. Adding engineered features (haversine distance, hour of day, day of week) would likely improve R² significantly.

#### 3. Iris Regression (`XGBoost_reg/xgb_reg_iris.py`)

**Dataset:** Scikit-learn Iris dataset — classic benchmark for testing implementations

### Classification

#### 4. Customer Segmentation (`XGBoost_cla/xgb_cus_cla.py`)

**Dataset:** Kaggle customer segmentation dataset (`Train.csv`, `Test.csv`) — 4-class classification

**Features:** Gender, Ever_Married, Age, Graduated, Profession, Work_Experience, Spending_Score, Family_Size, Var_1

**Missing Values:**

| Feature | Missing Count | % Missing |
|---------|---------------|-----------|
| Work_Experience | 829 | ~10.3% |
| Family_Size | 335 | ~4.2% |
| Ever_Married | 140 | ~1.7% |
| Profession | 124 | ~1.5% |
| Graduated | 78 | ~1.0% |
| Var_1 | 76 | ~0.9% |

**Results:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.39 | 0.38 | 0.38 | 391 |
| 1 | 0.40 | 0.34 | 0.37 | 369 |
| 2 | 0.50 | 0.58 | 0.54 | 380 |
| 3 | 0.67 | 0.69 | 0.68 | 474 |
| **Overall Accuracy** | | | **0.51** | **1,614** |

**Interpretation:** 51% accuracy on a 4-class problem (random baseline = 25%) shows the model has learned meaningful patterns, but performance is moderate. Class 3 is easiest to identify (F1 = 0.68) while classes 0 and 1 are frequently confused with each other (similar customer profiles). The significant missing data in Work_Experience (10.3%) hurts — XGBoost handles missing values natively, but having 10% of a key feature missing still limits predictive power. Potential improvements: feature engineering from existing features, dimensionality reduction, or trying CatBoost for its superior categorical handling.

---

## Usage

```bash
# Regression experiments
cd XGBoost_reg
python3 xgb_reg_laptop.py     # Outputs metrics + feature importance chart
python3 xgb_reg_uber.py       # Outputs metrics + feature importance chart
python3 xgb_reg_iris.py       # Outputs metrics

# Classification experiment
cd ../XGBoost_cla
python3 xgb_cus_cla.py        # Outputs confusion matrix + classification report
```

---

## References

- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
