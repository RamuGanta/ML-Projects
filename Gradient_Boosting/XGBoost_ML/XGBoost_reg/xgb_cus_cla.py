import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("train.csv")

df.head()

# Check for missing values
print(df.isnull().sum())

# Identify column types
numerical_features = ['Age','Work_Experience', 'Family_Size']
categorical_features = ['Gender','Ever_Married','Graduated','Profession','Var_1']

# Replace null values in numerical columns with the mean
for col in numerical_features:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing values for categorical columns with mode
for col in categorical_features:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
# Encode the target variable
le = LabelEncoder()
df['Segmentation'] = le.fit_transform(df['Segmentation'])

# Define features and target -- # target is the column 'Segmentation' - A, B , C & D
X = df.drop(['Segmentation', 'ID'], axis=1)
y = df['Segmentation']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
clf_pipeline.fit(X_train, y_train)

y_pred = clf_pipeline.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
