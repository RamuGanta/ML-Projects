# ML Projects

A hands-on collection of machine learning experiments comparing **three gradient boosting frameworks** — XGBoost, LightGBM, and CatBoost — across classification and regression tasks on real-world Kaggle datasets.

**Author:** [Ramu Ganta](https://github.com/RamuGanta) · [LinkedIn](https://www.linkedin.com/in/ramgan333729/)

---

## Project Structure

```
ML-Projects/
├── Gradient_Boosting/
│   ├── XGBoost_ML/              # XGBoost experiments
│   │   ├── XGBoost_reg/         # Regression: laptop prices, uber fares, iris
│   │   └── XGBoost_cla/         # Classification: customer segmentation
│   ├── LightGBM/                # LightGBM experiments
│   │   ├── reg/                 # Regression: laptop prices
│   │   └── cla/                 # Classification: bank churn
│   ├── CatBoost/                # CatBoost experiments
│   │   ├── reg/                 # Regression: Melbourne housing prices
│   │   └── cla/                 # Classification: bank churn
│   ├── Credit_Data.csv
│   └── test.py
└── README.md
```

---

## Results Summary

### Regression — Model Comparison

| Model | Dataset | R² Score | RMSE |
|-------|---------|----------|------|
| XGBoost | Laptop Prices | 0.728 | 371.58 |
| **LightGBM** | **Laptop Prices** | **0.800** | **321.16** |
| CatBoost | Melbourne Housing | 0.860 | 236,505 |
| XGBoost | Uber Fares | 0.692 | 5.66 |

LightGBM outperformed XGBoost on the same laptop dataset with an R² improvement from 0.728 to 0.800.

### Classification — Model Comparison

| Model | Dataset | Accuracy | Churn F1 |
|-------|---------|----------|----------|
| LightGBM | Bank Churn | 86% | 0.58 |
| **CatBoost** | **Bank Churn** | **87%** | **0.61** |
| XGBoost | Customer Segmentation | 51% | 0.49 avg |

CatBoost achieved the highest churn detection accuracy (87%) with native categorical feature handling and early stopping.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Boosting Frameworks | XGBoost, LightGBM, CatBoost |
| ML Ecosystem | scikit-learn, pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Datasets | Kaggle (laptop prices, uber fares, Melbourne housing, bank churn, customer segmentation) |

---

## Getting Started

```bash
git clone https://github.com/RamuGanta/ML-Projects.git
cd ML-Projects/Gradient_Boosting

# Create virtual environment
python3 -m venv xbvenv
source xbvenv/bin/activate

# Install dependencies
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn
```

Then navigate to any project folder and run the script:

```bash
cd XGBoost_ML/XGBoost_reg
python3 xgb_reg_laptop.py
```

---

## Key Takeaways

- **LightGBM** trains fastest and handles large datasets efficiently with histogram-based learning
- **CatBoost** excels with categorical features out of the box — no manual encoding needed
- **XGBoost** provides strong baseline performance with excellent feature importance visualization
- Geographic features (lat/long) dominated Uber fare prediction, while RAM was the strongest predictor for laptop prices
- Early stopping in CatBoost prevented overfitting at just 87 iterations vs the full 2,000

---

## Related Projects

- [InterviewAI-Backend](https://github.com/RamuGanta/InterviewAI-Backend) — FastAPI + OpenAI GPT-4 mock interview API
- [InterviewAI-Frontend](https://github.com/RamuGanta/InterviewAI-Frontend) — Streamlit UI for AI-powered interviews
- [Ramu_AI_ML](https://github.com/RamuGanta/Ramu_AI_ML) — AI/ML production systems & experiments

---

## Author

**Ramu Ganta** — [LinkedIn](https://www.linkedin.com/in/ramgan333729/) · [GitHub](https://github.com/RamuGanta)
