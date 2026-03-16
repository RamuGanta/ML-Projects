# Gradient Boosting Experiments

Comparative study of three industry-standard gradient boosting frameworks on real-world datasets.

## Frameworks Covered

| Framework | Strengths | Used For |
|-----------|-----------|----------|
| **XGBoost** | Regularization, feature importance | Laptop prices, Uber fares, Iris, Customer segmentation |
| **LightGBM** | Speed, histogram-based splits, memory efficient | Laptop prices, Bank churn |
| **CatBoost** | Native categorical support, early stopping | Melbourne housing, Bank churn |

## Folder Structure

```
Gradient_Boosting/
├── XGBoost_ML/          # XGBoost classification & regression
│   ├── XGBoost_reg/     # 3 regression tasks + 1 classification
│   └── XGBoost_cla/     # Customer segmentation (4 classes)
├── LightGBM/            # LightGBM classification & regression
│   ├── reg/             # Laptop price prediction
│   └── cla/             # Bank churn prediction
├── CatBoost/            # CatBoost classification & regression
│   ├── reg/             # Melbourne housing price prediction
│   └── cla/             # Bank churn prediction
├── Credit_Data.csv      # Shared dataset
└── test.py              # Quick test script
```

## Quick Results

**Best regression:** CatBoost on Melbourne housing — R² = 0.86

**Best classification:** CatBoost on bank churn — 87% accuracy, with early stopping at iteration 86

**Cross-framework comparison (laptop prices):** LightGBM (R² = 0.80) beat XGBoost (R² = 0.728) on the same dataset

## Setup

```bash
python3 -m venv xbvenv
source xbvenv/bin/activate
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn
```
