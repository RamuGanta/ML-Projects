# CatBoost Experiments

CatBoost (Categorical Boosting) implementations using Yandex's gradient boosting framework with native categorical feature support.

## Projects

### Regression (`reg/`)

| Script | Dataset | Task | R² Score | RMSE |
|--------|---------|------|----------|------|
| `catboost_reg.py` | `melb_data.csv` | Melbourne housing price prediction | **0.860** | 236,505 |

The highest R² score across all boosting frameworks in this repository.

**Training details:**
- Dataset: 13,580 rows, 21 columns
- Iterations: 2,000 (best at iteration 1,999)
- Training loss dropped from 622K to 112K over training
- Validation loss converged at ~236K

### Classification (`cla/`)

| Script | Dataset | Task | Accuracy |
|--------|---------|------|----------|
| `cust_churn.py` | `Bank Customer...urn Prediction.csv` | Bank churn prediction | **87%** |

Best churn classifier across all three frameworks. Two approaches compared:

| Approach | Accuracy | Churn Precision | Churn F1 |
|----------|----------|-----------------|----------|
| **Direct CatBoost** (cat feature names) | **87%** | **0.82** | **0.61** |
| Pipeline CatBoost (cat feature indices) | 86% | 0.74 | 0.59 |

**Key findings:**
- Early stopping triggered at iteration 86 (50-iteration patience) — prevented overfitting
- Direct CatBoost with named categorical features outperformed the pipeline approach
- 10,000 rows, 12 columns, zero missing values
- Features: credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary

## Why CatBoost?

- **Native categorical handling** — no need for manual label/one-hot encoding
- **Ordered boosting** — reduces prediction shift (overfitting on training data)
- **Built-in early stopping** — automatically stops when validation metric plateaus
- **GPU training support** — scales to larger datasets

## Usage

```bash
cd reg && python3 catboost_reg.py     # ~20 seconds, outputs RMSE + R²
cd ../cla && python3 cust_churn.py    # Prints two classification reports
```

Note: CatBoost generates a `catboost_info/` folder during training with training logs and metadata. This is auto-generated and can be safely ignored or added to `.gitignore`.
