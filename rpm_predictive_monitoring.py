import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# AI-Powered Remote Patient Monitoring - BME Portfolio Project #4
# Author: Michael Lozada-Longog
# Focus: Personalized monitoring for Pacific Islander & military populations
# Specialties: Remote Patient Monitoring, Biological Systems Modeling, AI for Health Equity
# =====================================================

print("🚀 Starting AI-Powered RPM Predictive Monitoring Project...\n")

# Simulated RPM dataset (vitals + demographics - extendable with real wearable data)
# In production: integrate with device APIs (e.g., Fitbit, Apple Health, military wearables)
np.random.seed(42)
n = 5000
data = {
    'PatientID': range(1, n+1),
    'Age': np.random.randint(18, 75, n),
    'Ethnicity': np.random.choice(['Pacific Islander', 'Military', 'Other'], n, p=[0.4, 0.4, 0.2]),
    'Glucose': np.random.normal(110, 25, n),
    'BloodPressure': np.random.normal(130, 15, n),
    'HeartRate': np.random.normal(78, 12, n),
    'BMI': np.random.normal(29, 6, n),
    'StressLevel': np.random.randint(1, 11, n),  # Military-relevant
    'ActivityLevel': np.random.choice(['Low', 'Moderate', 'High'], n),
    'FlareRisk': np.random.choice([0, 1], n, p=[0.7, 0.3])  # Target: high risk of flare (diabetes/HTN)
}

df = pd.DataFrame(data)

# Feature Engineering for biological systems modeling
df['Glucose_Insulin_Index'] = df['Glucose'] * (df['BMI'] / 25)  # Simplified proxy
df['CardioRiskScore'] = (df['BloodPressure'] + df['HeartRate'] * 0.5 + df['BMI'] * 2) / 3

print(f"✅ Simulated RPM Dataset Loaded — {len(df):,} patient records\n")
print("Unique Ethnicities:", df['Ethnicity'].unique())

# Modeling
features = ['Age', 'Glucose', 'BloodPressure', 'HeartRate', 'BMI', 'StressLevel', 'Glucose_Insulin_Index', 'CardioRiskScore']
X = pd.get_dummies(df[features + ['Ethnicity', 'ActivityLevel']], drop_first=True)
y = df['FlareRisk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"✅ Random Forest Accuracy for Flare Risk Prediction: {accuracy_score(y_test, y_pred):.1%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance (key for AI-integrated device explainability)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Top Features for RPM Flare Risk Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('rpm_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Interactive Dashboard-style Visualization (for founder demo)
fig = px.scatter(df, x='Glucose', y='BloodPressure', color='FlareRisk',
                 hover_data=['Ethnicity', 'StressLevel'],
                 title='Remote Patient Monitoring: Glucose vs BP by Flare Risk')
fig.write_html('rpm_interactive_dashboard.html')
print("📊 Interactive RPM dashboard saved as: rpm_interactive_dashboard.html")

# Save enriched dataset
df.to_csv('rpm_monitoring_cleaned.csv', index=False)
print("💾 Enriched RPM dataset saved as: rpm_monitoring_cleaned.csv")
print("\n🎉 Project 4 Completed! This demonstrates AI-driven RPM for personalized Pacific Islander & military health outcomes.")