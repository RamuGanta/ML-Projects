# XGBoost Experiments

XGBoost (Extreme Gradient Boosting) implementations for both regression and classification tasks.

## Projects

### Regression (`XGBoost_reg/`)

| Script | Dataset | Task | R² Score | RMSE |
|--------|---------|------|----------|------|
| `xgb_reg_laptop.py` | `laptop_price.csv` | Laptop price prediction | 0.728 | 371.58 |
| `xgb_reg_uber.py` | `uber.csv` | Uber fare prediction | 0.692 | 5.66 |
| `xgb_reg_iris.py` | Iris (sklearn) | Iris regression | — | — |

**Key findings:**
- **Laptop prices:** RAM is the dominant feature (importance ~0.40), followed by TypeName (~0.32)
- **Uber fares:** Geographic coordinates (dropoff/pickup longitude & latitude) drive predictions, with dropoff_longitude at ~0.41 importance

### Classification (`XGBoost_cla/`)

| Script | Dataset | Task | Accuracy |
|--------|---------|------|----------|
| `xgb_cus_cla.py` | `Train.csv` / `Test.csv` | Customer segmentation (4 classes) | 51% |

**Key findings:**
- Dataset had significant missing values (Work_Experience: 829, Family_Size: 335)
- Class 3 was easiest to predict (F1 = 0.68), Class 1 hardest (F1 = 0.37)
- Multi-class segmentation is inherently harder than binary classification

## Usage

```bash
cd XGBoost_reg
python3 xgb_reg_laptop.py    # Generates feature importance chart + metrics

cd ../XGBoost_cla
python3 xgb_cus_cla.py       # Prints confusion matrix + classification report
```
