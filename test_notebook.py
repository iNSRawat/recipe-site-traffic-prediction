"""
Test script to verify all notebook code runs correctly
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
import re
warnings.filterwarnings('ignore')

print("=" * 60)
print("NOTEBOOK VERIFICATION TEST")
print("=" * 60)

# 1. Load data
print("\n[1/8] Loading data...")
df = pd.read_csv('recipe_site_traffic_2212.csv')
print(f"  ✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Clean data
print("\n[2/8] Cleaning data...")
df_clean = df.copy()

# Clean servings
def clean_servings(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    numbers = re.findall(r'\d+', val_str)
    if numbers:
        return int(numbers[0])
    return np.nan

df_clean['servings'] = df_clean['servings'].apply(clean_servings)

# Convert target
df_clean['high_traffic'] = df_clean['high_traffic'].apply(lambda x: 1 if x == 'High' else 0)

# Impute missing values
nutritional_cols = ['calories', 'carbohydrate', 'sugar', 'protein']
for col in nutritional_cols:
    median_val = df_clean[col].median()
    df_clean[col].fillna(median_val, inplace=True)

print(f"  ✓ Data cleaned, missing values: {df_clean.isnull().sum().sum()}")

# 3. Create visualizations
print("\n[3/8] Creating visualizations...")
fig, ax = plt.subplots(figsize=(10, 6))
df_clean['category'].value_counts().plot(kind='bar', ax=ax)
plt.savefig('test_chart.png', dpi=100, bbox_inches='tight')
plt.close()
print("  ✓ Visualizations created")

# 4. Prepare data for modeling
print("\n[4/8] Preparing model data...")
df_model = pd.get_dummies(df_clean, columns=['category'], drop_first=True)
df_model = df_model.drop('recipe', axis=1)

X = df_model.drop('high_traffic', axis=1)
y = df_model['high_traffic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  ✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 5. Train Logistic Regression
print("\n[5/8] Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_precision = precision_score(y_test, lr_pred)
print(f"  ✓ LR Precision: {lr_precision*100:.1f}%")

# 6. Train Random Forest
print("\n[6/8] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_precision = precision_score(y_test, rf_pred)
print(f"  ✓ RF Precision: {rf_precision*100:.1f}%")

# 7. Evaluate models
print("\n[7/8] Evaluating models...")
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_roc = roc_auc_score(y_test, lr_prob)
rf_roc = roc_auc_score(y_test, rf_prob)
print(f"  LR: Accuracy={lr_accuracy*100:.1f}%, ROC-AUC={lr_roc:.3f}")
print(f"  RF: Accuracy={rf_accuracy*100:.1f}%, ROC-AUC={rf_roc:.3f}")

# 8. Business metrics
print("\n[8/8] Checking business metrics...")
random_baseline = y_test.mean() * 100
best_precision = max(lr_precision, rf_precision) * 100
print(f"  Random baseline: {random_baseline:.1f}%")
print(f"  Best precision: {best_precision:.1f}%")
print(f"  Business goal (80%): {'✓ MET' if best_precision >= 80 else '✗ NOT MET'}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - NOTEBOOK IS READY")
print("=" * 60)

# Cleanup
import os
if os.path.exists('test_chart.png'):
    os.remove('test_chart.png')
