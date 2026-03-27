import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# =====================================================
# Healthcare Admissions Analysis - BME Portfolio Project #2
# Author: Michael Lozada-Longog
# Dataset: Synthetic Healthcare Admissions (~55,500 patients)
# =====================================================

print("🚀 Starting Healthcare Admissions Analysis...\n")

# 1. Load the dataset
df = pd.read_csv('healthcare_dataset.csv')

print("✅ Healthcare Dataset Loaded Successfully!")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# === BASIC DATA QUALITY CHECKS (New - Recommended) ===
print("=== Data Quality Overview ===")
print(f"Missing values:\n{df.isnull().sum()}\n")

print("Unique Medical Conditions:", sorted(df['Medical Condition'].unique()))
print("Unique Admission Types:", sorted(df['Admission Type'].unique()))
print("Unique Test Results:", sorted(df['Test Results'].unique()))
print(f"Date range: {df['Date of Admission'].min()} to {df['Discharge Date'].max()}\n")

# 2. Calculate Length of Stay (core feature engineering)
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Length of Stay (days)'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

print("--- Length of Stay Statistics ---")
print(df['Length of Stay (days)'].describe().round(2))

# 3. In-depth Analysis by Medical Condition
condition_analysis = df.groupby('Medical Condition').agg({
    'Length of Stay (days)': ['mean', 'median', 'std', 'min', 'max', 'count'],
    'Billing Amount': ['mean', 'median', 'std'],
    'Age': ['mean', 'std']
}).round(2)

print("\n--- Detailed Analysis by Medical Condition ---")
print(condition_analysis)

# 4. Test Results Distribution by Condition (%)
print("\n--- Test Results by Medical Condition (%) ---")
crosstab = pd.crosstab(df['Medical Condition'], df['Test Results'], normalize='index').round(3) * 100
print(crosstab)

# 5. Statistical Test: ANOVA on Length of Stay by Medical Condition
print("\n=== Statistical Test: One-way ANOVA on Length of Stay ===")
groups = [df[df['Medical Condition'] == cond]['Length of Stay (days)'] 
          for cond in df['Medical Condition'].unique()]
anova = stats.f_oneway(*groups)
print(f"F-statistic: {anova.statistic:.4f}")
print(f"p-value: {anova.pvalue:.6f}")
if anova.pvalue < 0.05:
    print("✅ Significant differences in average Length of Stay across conditions.")
else:
    print("No statistically significant differences in LOS across conditions.")

# 6. Enhanced Visualizations (4 → 6 panels for more depth)
plt.figure(figsize=(18, 14))

plt.subplot(3, 2, 1)
sns.barplot(data=df, x='Medical Condition', y='Length of Stay (days)', errorbar=None)
plt.title('Average Length of Stay by Medical Condition')
plt.xticks(rotation=45)

plt.subplot(3, 2, 2)
sns.boxplot(data=df, x='Medical Condition', y='Billing Amount')
plt.title('Billing Amount Distribution by Medical Condition')
plt.xticks(rotation=45)

plt.subplot(3, 2, 3)
sns.boxplot(data=df, x='Admission Type', y='Length of Stay (days)')
plt.title('Length of Stay by Admission Type')

plt.subplot(3, 2, 4)
sns.histplot(data=df, x='Age', hue='Medical Condition', multiple='stack', bins=25, alpha=0.7)
plt.title('Age Distribution by Medical Condition')

plt.subplot(3, 2, 5)
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numeric Features)')

plt.subplot(3, 2, 6)
sns.countplot(data=df, x='Test Results', hue='Medical Condition')
plt.title('Test Results Count by Medical Condition')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('healthcare_visualizations_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Save enriched dataset
df.to_csv('healthcare_cleaned.csv', index=False)

print("\n🎉 Analysis Completed Successfully!")
print("📊 Visualization saved as: healthcare_visualizations_enhanced.png")
print("💾 Enriched dataset saved as: healthcare_cleaned.csv")
print(f"Total patients analyzed: {len(df):,}")
