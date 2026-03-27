import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# Pima Indian Diabetes Analysis - Biomedical Engineering Portfolio Project
# Author: Your Name
# Date: March 2026
# Goal: Clean clinical data, engineer features, and explore diabetes risk factors
# =====================================================

# Load the standard Pima Diabetes dataset
df = pd.read_csv('diabetes.csv')

print("✅ Pima Indian Diabetes Dataset Loaded")
print(f"Shape: {df.shape} (rows, columns)\n")
print("Columns:", df.columns.tolist())

# 1. Handle medically impossible zero values → NaN
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print("\nMissing values after cleaning zeros:")
print(df[zero_cols].isnull().sum())

# 2. Feature Engineering: BMI Categories (common in clinical BME work)
def categorize_bmi(bmi):
    if pd.isna(bmi):
        return 'Unknown'
    elif bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMICategory'] = df['BMI'].apply(categorize_bmi)

print("\nBMI Category Distribution:")
print(df['BMICategory'].value_counts())

# 3. Summary Statistics (key for clinical interpretation)
print("\nClinical Summary Statistics:")
print(df[['Glucose', 'BMI', 'Insulin', 'BloodPressure', 'Age']].describe())

print("\nDiabetes Prevalence:")
print(df['Outcome'].value_counts(normalize=True).round(3))

# 4. Visualizations
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='Glucose', y='Insulin', hue='Outcome', alpha=0.7, palette='viridis')
plt.title('Glucose vs Insulin by Diabetes Outcome')

plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Outcome', y='Glucose')
plt.title('Glucose Levels by Outcome')

plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='Outcome', y='BMI')
plt.title('BMI by Outcome')

plt.subplot(2, 2, 4)
sns.countplot(data=df, x='BMICategory', hue='Outcome')
plt.title('BMI Categories vs Diabetes Outcome')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('diabetes_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# Save cleaned version
df.to_csv('diabetes_cleaned.csv', index=False)
print("\n✅ Cleaned dataset saved as 'diabetes_cleaned.csv'")
print("✅ Visualization saved as 'diabetes_visualizations.png'")