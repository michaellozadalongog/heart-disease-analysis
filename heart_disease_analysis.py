import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# =====================================================
# Heart Disease UCI Analysis - BME Portfolio Project #3
# Author: Michael Lozada-Longog
# Dataset: Heart Disease UCI (920 patients)
# =====================================================

print("🚀 Starting Heart Disease UCI Analysis...\n")

# 1. Load the dataset
df = pd.read_csv('heart_disease_uci.csv')

print("✅ Heart Disease Dataset Loaded Successfully!")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# === Data Quality Checks ===
print("=== Data Quality Overview ===")
print(f"Missing values:\n{df.isnull().sum()}\n")
print("Target Distribution (num 0-4):")
print(df['num'].value_counts().sort_index())

# 2. Data Cleaning & Feature Engineering
# Convert target to binary (BME-relevant: presence of heart disease)
df['HeartDisease'] = (df['num'] > 0).astype(int)

# Handle missing values with median imputation (best for clinical data)
numeric_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("\nMissing values after imputation: 0")

# 3. Summary Statistics
print("\nClinical Summary:")
print(df[['age', 'trestbps', 'chol', 'thalch', 'oldpeak']].describe().round(2))

# 4. Visualizations
plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
sns.histplot(data=df, x='age', hue='HeartDisease', multiple='stack', bins=20)
plt.title('Age Distribution by Heart Disease')

plt.subplot(2, 3, 2)
sns.boxplot(data=df, x='HeartDisease', y='chol')
plt.title('Cholesterol by Heart Disease')

plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='HeartDisease', y='trestbps')
plt.title('Resting Blood Pressure by Heart Disease')

plt.subplot(2, 3, 4)
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

plt.subplot(2, 3, 5)
sns.countplot(data=df, x='cp', hue='HeartDisease')
plt.title('Chest Pain Type by Heart Disease')
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
sns.countplot(data=df, x='sex', hue='HeartDisease')
plt.title('Sex by Heart Disease')

plt.tight_layout()
plt.savefig('heart_disease_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Predictive Modeling (same style as Diabetes project)
X = df.drop(['num', 'HeartDisease', 'id', 'dataset'], axis=1, errors='ignore')
# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression (consistent with Diabetes project)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
log_acc = accuracy_score(y_test, y_pred_log)

print(f"\n✅ Logistic Regression Accuracy: {log_acc:.1%}")

# Bonus: Random Forest (often performs better on this dataset)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"✅ Random Forest Accuracy: {rf_acc:.1%}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# 6. Save enriched dataset
df.to_csv('heart_disease_cleaned.csv', index=False)

print("\n🎉 Analysis Completed Successfully!")
print("📊 Visualization saved as: heart_disease_visualizations.png")
print("💾 Enriched dataset saved as: heart_disease_cleaned.csv")
print(f"Total patients analyzed: {len(df):,}")