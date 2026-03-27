import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# If the file is in your Downloads, use the full path like:
# df = pd.read_csv(r'/Users/yourname/Downloads/heart_disease_uci.csv')
df = pd.read_csv('/Users/michellelongog/Downloads/heart_disease_uci.csv')

# 2. Basic Data Overview
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info (Check for missing values) ---")
print(df.info())

# 3. Simple Summary Statistics for BME metrics
print("\n--- Statistics for Blood Pressure and Cholesterol ---")
print(df[['trestbps', 'chol']].describe())

# 4. Check the Target Variable (Heart Disease Presence)
# 'num' usually indicates severity from 0 (none) to 4 (severe)
print("\n--- Heart Disease Diagnosis Counts ---")
print(df['num'].value_counts())
