# Gradient Boosting Experiments

Comparative implementations of XGBoost, LightGBM, and CatBoost on real-world datasets, organized by framework and task type (regression vs classification).

---

## How Gradient Boosting Works

Gradient Boosting builds an ensemble of decision trees **sequentially**. Unlike Random Forest where trees are built independently and averaged, here each tree learns from the mistakes of all previous trees combined.

### The Intuition

Imagine predicting house prices. Your first tree gives rough estimates — it gets the general range right but makes errors. The second tree doesn't predict house prices directly; instead, it predicts **the errors** the first tree made. The third tree predicts the errors that remain after combining the first two trees. After hundreds of these corrections, the ensemble's predictions become highly accurate.

### Mathematically

The model at round `m` is:

```
Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)
```

where:
- `Fₘ₋₁(x)` is the current ensemble prediction
- `hₘ(x)` is the new tree fitted to the **negative gradient** of the loss function
- `η` is the learning rate (shrinkage), controlling how much each tree contributes

The negative gradient for common loss functions:

| Loss | Task | Gradient (what the next tree learns) |
|------|------|--------------------------------------|
| `½(y - F)²` | Regression | `y - F` (the residual) |
| `-y·log(σ(F)) - (1-y)·log(1-σ(F))` | Binary classification | `y - σ(F)` (where σ is sigmoid) |
| Huber loss | Robust regression | Residual if small, sign if large |

### Why "Gradient" Boosting?

The "gradient" refers to **gradient descent in function space**. Just as gradient descent updates model parameters by moving in the direction of steepest loss reduction, gradient boosting updates the ensemble by adding a tree that points in the direction of steepest loss reduction in the space of predictions.

---

## Framework Comparison on Same Tasks

### Laptop Price Prediction (Regression)

| Metric | XGBoost | LightGBM | Improvement |
|--------|---------|----------|-------------|
| R² | 0.728 | **0.800** | +9.9% |
| RMSE | 371.58 | **321.16** | -13.6% |
| MAE | 212.78 | **176.55** | -17.0% |

LightGBM's leaf-wise growth strategy finds better splits than XGBoost's level-wise approach on this dataset.

### Bank Churn Prediction (Classification)

| Metric | LightGBM | CatBoost | Improvement |
|--------|----------|----------|-------------|
| Accuracy | 86% | **87%** | +1% |
| Churn Precision | 0.74 | **0.82** | +10.8% |
| Churn F1 | 0.58 | **0.61** | +5.2% |

CatBoost's native categorical handling (country, gender) and ordered boosting gave it an edge, especially in churn precision.

---

## Folder Structure

```
Gradient_Boosting/
├── XGBoost_ML/          # XGBoost: 4 experiments (3 regression + 1 classification)
│   ├── XGBoost_reg/     # Laptop prices, Uber fares, Iris
│   └── XGBoost_cla/     # Customer segmentation (4-class)
├── LightGBM/            # LightGBM: 2 experiments
│   ├── reg/             # Laptop price prediction (R² = 0.80)
│   └── cla/             # Bank churn prediction (86% accuracy)
├── CatBoost/            # CatBoost: 2 experiments
│   ├── reg/             # Melbourne housing prices (R² = 0.86)
│   └── cla/             # Bank churn prediction (87% accuracy)
├── Credit_Data.csv
└── test.py
```

---

## Choosing the Right Framework

```
Is your data mostly categorical? ──► YES ──► CatBoost
         │
         NO
         │
Is your dataset large (>100K rows)? ──► YES ──► LightGBM
         │
         NO
         │
Do you need maximum control / ──► YES ──► XGBoost
fine-tuned regularization?
         │
         NO
         │
Start with XGBoost (most documented, largest community)
```

---

## Setup

```bash
python3 -m venv xbvenv
source xbvenv/bin/activate
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn
```
