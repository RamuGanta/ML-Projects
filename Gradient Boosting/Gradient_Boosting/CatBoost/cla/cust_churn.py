import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

# 1) Load + basic checks

df = pd.read_csv("Bank Customer Churn Prediction.csv")

print("Shape:", df.shape)
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# Drop identifier column
if "customer_id" in df.columns:
    df = df.drop(columns=["customer_id"])

# Define features/target
X = df.drop(columns=["churn"])
y = df["churn"]

# CatBoost categorical columns (by name)
cat_cols = ["country", "gender"]

# Convert categorical column names -> indices (needed for sklearn Pipeline because it becomes numpy)
cat_col_indices = [X.columns.get_loc(c) for c in cat_cols]


# 2) Split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3) A) Direct CatBoost (names work because X is a DataFrame)

direct_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_cols,          # names OK here
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

direct_model.fit(X_train, y_train, eval_set=(X_val, y_val))

y_pred_direct = direct_model.predict(X_val)
print("\n=== Direct CatBoost (DataFrame + cat feature NAMES) ===")
print(classification_report(y_val, y_pred_direct))



# 4) B) sklearn Pipeline CatBoost (use indices)

pipeline = Pipeline(steps=[
    ("classifier", CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_col_indices,  # indices REQUIRED in pipeline
        random_seed=42,
        verbose=0
    ))
])

pipeline.fit(X_train, y_train)

y_pred_pipe = pipeline.predict(X_val)
print("\n=== Pipeline CatBoost (cat feature INDICES) ===")
print(classification_report(y_val, y_pred_pipe))