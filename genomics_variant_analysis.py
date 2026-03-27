import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# AI-Driven Genomics Variant Analysis - BME Portfolio Project #5
# Author: Michael Lozada-Longog
# Focus: Personalized Genomics for Pacific Islander & Military Health
# =====================================================

print("🚀 Starting AI-Driven Genomics Variant Analysis (Project 5)...\n")

# Synthetic genomics dataset focused on Pacific Islander-relevant variants
np.random.seed(42)
n = 3000

data = {
    'PatientID': range(1, n+1),
    'Ethnicity': np.random.choice(['Pacific Islander', 'Military', 'Other'], n, p=[0.45, 0.35, 0.2]),
    'Age': np.random.randint(18, 70, n),
    'BMI': np.random.normal(29.5, 7, n).clip(15, 45),
    'Variant_TCF7L2': np.random.choice([0, 1, 2], n, p=[0.4, 0.45, 0.15]),   # Strong diabetes risk allele
    'Variant_APOE4': np.random.choice([0, 1, 2], n, p=[0.6, 0.35, 0.05]),    # Cardiovascular risk
    'Variant_HLA_B27': np.random.choice([0, 1], n, p=[0.85, 0.15]),           # Inflammatory conditions
    'PolygenicRiskScore': np.random.normal(0, 1, n),
    'DiabetesRisk': np.random.choice([0, 1], n, p=[0.65, 0.35])
}

df = pd.DataFrame(data)

print(f"✅ Genomics Dataset Loaded — {len(df):,} synthetic patient records\n")

# Feature Engineering for biological systems modeling
df['CombinedGeneticRisk'] = df['Variant_TCF7L2'] * 0.6 + df['Variant_APOE4'] * 0.4 + df['Variant_HLA_B27'] * 0.3

# Modeling
features = ['Age', 'BMI', 'Variant_TCF7L2', 'Variant_APOE4', 'Variant_HLA_B27', 'PolygenicRiskScore', 'CombinedGeneticRisk']
X = pd.get_dummies(df[features + ['Ethnicity']], drop_first=True)
y = df['DiabetesRisk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"✅ Random Forest Accuracy for Diabetes Risk Prediction: {accuracy_score(y_test, y_pred):.1%}")

# Feature Importance (critical for genomics AI tools)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Top Genetic & Clinical Features for Diabetes Risk (Pacific Islander Focus)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('genomics_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Genomics feature importance plot saved as: genomics_feature_importance.png")

# Save enriched dataset
df.to_csv('genomics_variants_cleaned.csv', index=False)
print("💾 Enriched genomics dataset saved as: genomics_variants_cleaned.csv")

print("\n🎉 Project 5 Completed! Demonstrates AI-driven genomics for personalized Pacific Islander health outcomes.")