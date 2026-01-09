import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('recipe_site_traffic_2212.csv')

# Print basic info
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("FIRST 10 ROWS")
print("=" * 50)
print(df.head(10))

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("UNIQUE VALUES PER COLUMN")
print("=" * 50)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
    if df[col].dtype == 'object' or df[col].nunique() < 20:
        print(f"  Values: {df[col].unique()[:10].tolist()}")

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY - NUMERIC")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY - CATEGORICAL")
print("=" * 50)
print(df.describe(include='object'))

print("\n" + "=" * 50)
print("TARGET VARIABLE (high_traffic)")
print("=" * 50)
print(df['high_traffic'].value_counts(dropna=False))
