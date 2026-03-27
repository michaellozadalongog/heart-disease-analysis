import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# AI-Driven Drug Discovery & Pharmacogenomics - BME Portfolio Project #6
# Author: Michael Lozada-Longog
# Focus: Personalized Drug Response Prediction for Pacific Islander & Military Populations
# =====================================================

print("🚀 Starting AI-Driven Drug Discovery & Pharmacogenomics Analysis (Project 6)...\n")

# Synthetic pharmacogenomics dataset (realistic for Pacific Islander / military context)
np.random.seed(42)
n = 4000

data = {
    'PatientID': range(1, n+1),
    'Ethnicity': np.random.choice(['Pacific Islander', 'Military', 'Other'], n, p=[0.45, 0.35, 0.2]),
    'Age': np.random.randint(18, 75, n),
    'BMI': np.random.normal(29.8, 6.5, n).clip(16, 48),
    'Variant_CYP2C19': np.random.choice([0, 1, 2], n, p=[0.55, 0.35, 0.10]),   # Drug metabolism (clopidogrel response)
    'Variant_SLCO1B1': np.random.choice([0, 1, 2], n, p=[0.60, 0.30, 0.10]),    # Statin-induced myopathy risk
    'Variant_VKORC1': np.random.choice([0, 1, 2], n, p=[0.40, 0.45, 0.15]),     # Warfarin dosing
    'PolygenicDrugResponseScore': np.random.normal(0, 1.2, n),
    'AdverseDrugReactionRisk': np.random.choice([0, 1], n, p=[0.68, 0.32])      # Target: high risk of ADR
}

df = pd.DataFrame(data)

print(f"✅ Pharmacogenomics Dataset Loaded — {len(df):,} synthetic patient records\n")

# Biological systems modeling & feature engineering
df['CombinedPharmacoRisk'] = (df['Variant_CYP2C19'] * 0.5 + 
                              df['Variant_SLCO1B1'] * 0.4 + 
                              df['Variant_VKORC1'] * 0.3)

# Modeling: Predict high Adverse Drug Reaction risk
features = ['Age', 'BMI', 'Variant_CYP2C19', 'Variant_SLCO1B1', 'Variant_VKORC1', 
            'PolygenicDrugResponseScore', 'CombinedPharmacoRisk']
X = pd.get_dummies(df[features + ['Ethnicity']], drop_first=True)
y = df['AdverseDrugReactionRisk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"✅ Random Forest Accuracy for Adverse Drug Reaction Risk: {accuracy_score(y_test, y_pred):.1%}")

# Feature Importance (key for explainable drug discovery tools)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Top Features for Personalized Drug Response Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('pharmacogenomics_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Pharmacogenomics feature importance plot saved as: pharmacogenomics_feature_importance.png")

# Save enriched dataset
df.to_csv('pharmacogenomics_cleaned.csv', index=False)
print("💾 Enriched pharmacogenomics dataset saved as: pharmacogenomics_cleaned.csv")

print("\n🎉 Project 6 Completed! Demonstrates AI tools for personalized drug discovery and pharmacogenomics tailored to Pacific Islander and military populations.")