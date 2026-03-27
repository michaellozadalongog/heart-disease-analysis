import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# =====================================================
# Heart Disease UCI Analysis - BME Portfolio Project #3
# Author: Michael Lozada-Longog
# Dataset: Heart Disease UCI (920 patients)
# =====================================================

print("🚀 Starting Heart Disease UCI Analysis...\n")

df = pd.read_csv('heart_disease_uci.csv')
print(f"✅ Dataset Loaded — Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# Data Cleaning
df['HeartDisease'] = (df['num'] > 0).astype(int)
numeric_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Visualizations
plt.figure(figsize=(18, 12))
# ... (same 6 plots as before - kept for brevity) ...
plt.tight_layout()
plt.savefig('heart_disease_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# Modeling
X = pd.get_dummies(df.drop(['num', 'HeartDisease', 'id', 'dataset'], axis=1, errors='ignore'), drop_first=True)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"✅ Random Forest Accuracy: {accuracy_score(y_test, y_pred):.1%}")

# NEW: Feature Importance Plot (BME highlight!)
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('Top 10 Feature Importance - Random Forest (Heart Disease)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('heart_disease_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Feature importance plot saved as: heart_disease_feature_importance.png")

# Save cleaned data
df.to_csv('heart_disease_cleaned.csv', index=False)
print("💾 Enriched dataset saved as: heart_disease_cleaned.csv")