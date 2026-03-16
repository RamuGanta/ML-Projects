# ML Projects — Gradient Boosting Deep Dive

A comprehensive, hands-on study of **three industry-standard gradient boosting frameworks** — XGBoost, LightGBM, and CatBoost — applied to real-world regression and classification tasks. Each framework is implemented from scratch on Kaggle datasets with detailed evaluation, feature importance analysis, and cross-framework performance comparison.

**Author:** [Ramu Ganta](https://github.com/RamuGanta) · [LinkedIn](https://www.linkedin.com/in/ramgan333729/)

---

## What Is Gradient Boosting?

Gradient Boosting is a sequential ensemble technique that builds a strong predictive model by combining many weak learners (typically decision trees), where each new tree is trained to correct the residual errors of the previous ensemble.

### The Core Algorithm

Given a dataset with `n` samples and a differentiable loss function `L(y, F(x))`:

1. **Initialize** the model with a constant prediction:

   ```
   F₀(x) = argmin_γ Σ L(yᵢ, γ)
   ```

   For regression with squared error, this is simply the mean of the target values.

2. **For each boosting round** `m = 1, 2, ..., M`:

   a. Compute the **pseudo-residuals** (negative gradient of the loss):
   ```
   rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]  evaluated at F = Fₘ₋₁
   ```
   For squared error loss `L = ½(y - F)²`, this simplifies to `rᵢₘ = yᵢ - Fₘ₋₁(xᵢ)`, which is the literal residual.

   b. Fit a weak learner `hₘ(x)` (decision tree) to the pseudo-residuals `{(xᵢ, rᵢₘ)}`.

   c. Compute the **optimal step size** (learning rate × tree prediction):
   ```
   γₘ = argmin_γ Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ · hₘ(xᵢ))
   ```

   d. **Update** the model:
   ```
   Fₘ(x) = Fₘ₋₁(x) + η · γₘ · hₘ(x)
   ```
   where `η` is the learning rate (shrinkage parameter, typically 0.01–0.3).

3. **Output** the final model `F_M(x)`.

### Loss Functions

The choice of loss function determines what the model optimizes:

| Task | Loss Function | Formula | Pseudo-Residual |
|------|--------------|---------|-----------------|
| Regression | Squared Error (L2) | `½(y - F)²` | `y - F` |
| Regression | Absolute Error (L1) | `\|y - F\|` | `sign(y - F)` |
| Binary Classification | Log Loss | `-[y·log(p) + (1-y)·log(1-p)]` | `y - p` (where `p = sigmoid(F)`) |
| Multi-class | Softmax Cross-Entropy | `-Σ yₖ·log(pₖ)` | `yₖ - pₖ` per class |

### Key Hyperparameters

| Parameter | What It Controls | Too Low | Too High |
|-----------|-----------------|---------|----------|
| `n_estimators` | Number of boosting rounds | Underfitting | Overfitting, slow |
| `learning_rate` | Step size shrinkage (η) | Needs more trees | Overfitting |
| `max_depth` | Tree complexity | Underfitting | Overfitting, captures noise |
| `subsample` | Row sampling per tree | High variance | No regularization benefit |
| `colsample_bytree` | Feature sampling per tree | May miss important features | No regularization benefit |
| `min_child_weight` | Minimum samples in leaf | Overfitting | Underfitting |

The **learning rate and n_estimators trade off**: lower learning rate needs more trees but usually gives better generalization. A common strategy is to set a low learning rate (0.01–0.1) and use early stopping to find the optimal number of trees.

---

## Framework Comparison

### How They Differ

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| **Tree growth** | Level-wise (balanced) | Leaf-wise (best-first) | Symmetric (balanced) |
| **Split finding** | Exact or histogram | Histogram-based | Oblivious decision trees |
| **Categorical features** | Manual encoding required | Native (optimal split) | Native (ordered target stats) |
| **Regularization** | L1 + L2 on weights | L1 + L2 on weights | Ordered boosting (reduces overfitting) |
| **Missing values** | Learns optimal direction | Assigns to gain-maximizing side | Uses "min" or "max" treatment |
| **Speed** | Fast | Fastest (2–5× faster) | Moderate (but less tuning needed) |
| **Best for** | General purpose, Kaggle | Large datasets, speed-critical | Categorical-heavy data, small data |

### When to Use Each

**XGBoost** — Use when you need a reliable, well-documented baseline. Works well for structured/tabular data with mostly numerical features. Good when you want fine-grained control over regularization. Industry standard for Kaggle competitions and production systems.

**LightGBM** — Use when training speed matters or your dataset is large (100K+ rows). Its leaf-wise growth finds complex patterns faster but can overfit on small datasets. Excellent for high-cardinality categorical features with its native encoding. Preferred in production environments where retraining speed matters.

**CatBoost** — Use when your data has many categorical features (country, gender, product type). Its ordered target statistics encoding handles categoricals without leakage. Requires less hyperparameter tuning than XGBoost/LightGBM. Built-in early stopping and overfitting detection make it beginner-friendly while still powerful.

---

## Project Structure

```
ML-Projects/
├── Gradient_Boosting/
│   ├── XGBoost_ML/                    # XGBoost experiments
│   │   ├── XGBoost_reg/               # Regression: laptop, uber, iris
│   │   │   ├── xgb_reg_laptop.py
│   │   │   ├── xgb_reg_uber.py
│   │   │   ├── xgb_reg_iris.py
│   │   │   ├── xgb_cus_cla.py
│   │   │   ├── laptop_price.csv
│   │   │   └── uber.csv
│   │   └── XGBoost_cla/               # Classification: customer segmentation
│   │       ├── xgb_cus_cla.py
│   │       ├── Train.csv
│   │       └── Test.csv
│   ├── LightGBM/                      # LightGBM experiments
│   │   ├── reg/                        # Regression: laptop prices
│   │   │   ├── laptop_lgbm.py
│   │   │   └── laptop_price.csv
│   │   └── cla/                        # Classification: bank churn
│   │       ├── churn_lgbm.py
│   │       └── Bank Customer Churn Prediction.csv
│   ├── CatBoost/                      # CatBoost experiments
│   │   ├── reg/                        # Regression: Melbourne housing
│   │   │   ├── catboost_reg.py
│   │   │   └── melb_data.csv
│   │   └── cla/                        # Classification: bank churn
│   │       ├── cust_churn.py
│   │       └── Bank Customer Churn Prediction.csv
│   ├── Credit_Data.csv
│   └── test.py
└── README.md
```

---

## Results

### Regression Performance

| Framework | Dataset | Samples | Features | MAE | RMSE | R² Score |
|-----------|---------|---------|----------|-----|------|----------|
| XGBoost | Laptop Prices | 1,303 | 12 | 212.78 | 371.58 | 0.728 |
| **LightGBM** | **Laptop Prices** | **1,303** | **12** | **176.55** | **321.16** | **0.800** |
| CatBoost | Melbourne Housing | 13,580 | 20 | — | 236,505 | 0.860 |
| XGBoost | Uber Fares | — | 7 | 2.54 | 5.66 | 0.692 |

**Analysis:** On the same laptop price dataset, LightGBM outperformed XGBoost by a significant margin — R² improved from 0.728 to 0.800 (a 10% gain in explained variance). LightGBM's leaf-wise growth strategy likely captures more complex feature interactions, particularly between Ram and TypeName which together account for ~72% of feature importance. CatBoost achieved the highest absolute R² (0.86) on Melbourne housing, benefiting from native handling of categorical features like suburb names, council areas, and property types.

### Classification Performance

| Framework | Dataset | Samples | Classes | Accuracy | Macro F1 | Best Class F1 |
|-----------|---------|---------|---------|----------|----------|---------------|
| XGBoost | Customer Segmentation | 8,068 | 4 | 51% | 0.49 | 0.68 (class 3) |
| LightGBM | Bank Churn | 10,000 | 2 | 86% | 0.75 | 0.92 (no churn) |
| **CatBoost** | **Bank Churn** | **10,000** | **2** | **87%** | **0.77** | **0.92 (no churn)** |

**Analysis:** CatBoost edged out LightGBM on the same churn dataset (87% vs 86%). The key difference was churn precision — CatBoost achieved 0.82 vs LightGBM's 0.74, meaning fewer false alarms when predicting churn. CatBoost's ordered boosting and early stopping (triggered at iteration 86 with 50-round patience) helped prevent overfitting on the imbalanced dataset (~20% churn rate). The XGBoost customer segmentation task was a harder 4-class problem with significant missing data, explaining the lower accuracy.

### Feature Importance Insights

**Laptop Prices (XGBoost):** Ram dominates at ~0.40 importance, followed by TypeName at ~0.32. This makes intuitive sense — RAM capacity is the primary hardware differentiator in laptop pricing, while laptop category (Gaming, Ultrabook, Notebook) defines the price tier.

**Uber Fares (XGBoost):** Geographic coordinates dominate — dropoff_longitude (0.41), pickup_longitude (0.30), pickup_latitude (0.13). Fare is fundamentally a function of distance, and longitude/latitude encode distance traveled. Temporal features (pickup_datetime) had minimal impact (~0.02), suggesting base fare pricing dominates over surge pricing in this dataset.

---

## Getting Started

### Prerequisites

- Python 3.8+

### Setup

```bash
git clone https://github.com/RamuGanta/ML-Projects.git
cd ML-Projects/Gradient_Boosting

# Create virtual environment
python3 -m venv xbvenv
source xbvenv/bin/activate  # On Windows: xbvenv\Scripts\activate

# Install all dependencies
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn
```

### Run Any Experiment

```bash
# XGBoost regression
cd XGBoost_ML/XGBoost_reg && python3 xgb_reg_laptop.py

# LightGBM classification
cd ../../LightGBM/cla && python3 churn_lgbm.py

# CatBoost regression
cd ../../CatBoost/reg && python3 catboost_reg.py
```

Each script outputs evaluation metrics to the terminal and displays a feature importance chart via matplotlib.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Boosting Frameworks | XGBoost, LightGBM, CatBoost |
| ML Ecosystem | scikit-learn, pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Datasets | Kaggle (laptop prices, Uber fares, Melbourne housing, bank churn, customer segmentation, Iris) |

---

## References

- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Prokhorenkova, L. et al. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS*.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*.

---

## Author

**Ramu Ganta** — [LinkedIn](https://www.linkedin.com/in/ramgan333729/) · [GitHub](https://github.com/RamuGanta)
