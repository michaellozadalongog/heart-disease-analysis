import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# Healthcare Admissions Analysis - BME Portfolio Project #2
# Author: Michael Lozada-Longog
# Dataset: Synthetic Healthcare Admissions (Kaggle)
# =====================================================

# Load the dataset
df = pd.read_csv('healthcare_dataset.csv')

print("✅ Healthcare Dataset Loaded Successfully!")
print(f"Shape: {df.shape} (rows, columns)\n")
print("Columns:", df.columns.tolist())

# 1. Calculate Length of Stay
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Length of Stay (days)'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

print("\n--- Length of Stay Statistics ---")
print(df['Length of Stay (days)'].describe().round(2))

# 2. Analysis by Medical Condition
condition_analysis = df.groupby('Medical Condition').agg({
    'Length of Stay (days)': ['mean', 'median', 'count'],
    'Billing Amount': ['mean', 'median'],
    'Age': 'mean'
}).round(2)

print("\n--- Analysis by Medical Condition ---")
print(condition_analysis)

# 3. Test Results by Medical Condition
print("\n--- Test Results by Medical Condition (%) ---")
print(pd.crosstab(df['Medical Condition'], df['Test Results'], normalize='index').round(3))

# 4. Visualizations
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
sns.barplot(data=df, x='Medical Condition', y='Length of Stay (days)', errorbar=None)
plt.title('Average Length of Stay by Medical Condition')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Medical Condition', y='Billing Amount')
plt.title('Billing Amount by Medical Condition')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='Admission Type', y='Length of Stay (days)')
plt.title('Length of Stay by Admission Type')

plt.subplot(2, 2, 4)
sns.histplot(data=df, x='Age', hue='Medical Condition', multiple='stack', bins=20, alpha=0.7)
plt.title('Age Distribution by Medical Condition')

plt.tight_layout()
plt.savefig('healthcare_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# Save enriched dataset
df.to_csv('healthcare_cleaned.csv', index=False)

print("\n✅ Analysis completed!")
print("✅ Visualization saved as 'healthcare_visualizations.png'")
print("✅ Enriched dataset saved as 'healthcare_cleaned.csv'")