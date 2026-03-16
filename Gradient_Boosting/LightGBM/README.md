# LightGBM Experiments

LightGBM (Light Gradient Boosting Machine) implementations using Microsoft's histogram-based boosting framework.

## Projects

### Regression (`reg/`)

| Script | Dataset | Task | R² Score | RMSE |
|--------|---------|------|----------|------|
| `laptop_lgbm.py` | `laptop_price.csv` | Laptop price prediction | **0.800** | 321.16 |

Outperforms XGBoost (R² = 0.728) on the same laptop dataset — a 10% improvement in explained variance with faster training time.

**Metrics:**
- MAE: 176.55
- MSE: 103,145.13
- RMSE: 321.16
- R² Score: 0.800

### Classification (`cla/`)

| Script | Dataset | Task | Accuracy |
|--------|---------|------|----------|
| `churn_lgbm.py` | `Bank Customer...urn Prediction.csv` | Bank churn prediction | **86%** |

**Classification report:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| No churn (0) | 0.88 | 0.96 | 0.92 |
| Churn (1) | 0.74 | 0.48 | 0.58 |

The model is strong at identifying non-churners but conservative on churn prediction — typical class imbalance behavior (only ~20% churn rate).

## Why LightGBM?

- **Histogram-based splitting** — bins continuous features for faster training
- **Leaf-wise growth** — grows the leaf with the highest loss reduction (vs level-wise in XGBoost)
- **Memory efficient** — uses less memory than XGBoost on the same data

## Usage

```bash
cd reg && python3 laptop_lgbm.py
cd ../cla && python3 churn_lgbm.py
```
