import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# =====================================================
# Pima Indian Diabetes Analysis - BME Portfolio Project
# =====================================================

df = pd.read_csv('diabetes.csv')

print("✅ Pima Indian Diabetes Dataset Loaded")
print(f"Shape: {df.shape}\n")

# 1. Clean impossible zeros
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print("Missing values after cleaning:")
print(df[zero_cols].isnull().sum())

# 2. Impute missing values (median is often better for skewed clinical data)
imputer = SimpleImputer(strategy='median')
df[zero_cols] = imputer.fit_transform(df[zero_cols])

print("\nMissing values after imputation: 0")

# 3. Feature Engineering - BMI Categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMICategory'] = df['BMI'].apply(categorize_bmi)

# 4. Summary
print("\nDiabetes Prevalence:", df['Outcome'].value_counts(normalize=True).round(3))
print("\nClinical Summary:")
print(df[['Glucose', 'BMI', 'Insulin', 'BloodPressure', 'Age']].describe().round(2))

# 5. Visualizations
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='Glucose', y='Insulin', hue='Outcome', alpha=0.7)
plt.title('Glucose vs Insulin by Outcome')

plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Outcome', y='Glucose')
plt.title('Glucose by Diabetes Outcome')

plt.subplot(2, 2, 3)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

plt.subplot(2, 2, 4)
sns.countplot(data=df, x='BMICategory', hue='Outcome')
plt.title('BMI Categories vs Outcome')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('diabetes_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Simple Predictive Model (Logistic Regression)
X = df.drop(['Outcome', 'BMICategory'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Logistic Regression Accuracy: {accuracy:.1%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save cleaned data
df.to_csv('diabetes_cleaned.csv', index=False)
print("\n✅ Files saved: diabetes_cleaned.csv and diabetes_visualizations.png")