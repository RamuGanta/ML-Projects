import pandas as pd

df = pd.read_csv("Credit_Data 2.csv")

print(df["age"].isnull().sum())