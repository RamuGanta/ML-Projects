# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


import warnings
warnings.filterwarnings("ignore")


# Load the Dataset
df = pd.read_csv("Credit_Data.csv", encoding='latin-1') 
df.head()

# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# dropping rows with missing values and factorize categorical columns.
df = df.dropna()

# Encodng categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

# Confirm no null values and all columns are numeric
# print(df.dtypes)
# print(df.isnull().sum().sum())

# set X and y
X = df.drop(columns=['Rating'])
y = df['Rating']

# Split the data into training and testing sets for model evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize the XGB Regressor

model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make Predictions
# Use the trained model to predict prices on the test set.
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
# Evaluate the model using common regression metrics: MAE, MSE, RMSE, and R^2 Score.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)

# Visualize which features had the most influence on predictions using feature importances.
plt.figure(figsize=(12, 6))
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()