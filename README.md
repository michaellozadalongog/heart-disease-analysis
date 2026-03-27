# Pima Indian Diabetes Risk Analysis

**Biomedical Engineering Portfolio Project**

Exploratory data analysis and feature engineering on the classic Pima Indians Diabetes dataset.

## Key Highlights
- Handled medically invalid zero values (Glucose, BMI, Insulin, etc.) as missing data
- Created clinical BMI categories (Underweight / Normal / Overweight / Obese)
- Visualized relationships between Glucose, Insulin, BMI and diabetes outcome
- Generated cleaned dataset and high-resolution plots

## Technologies
- Python, Pandas, NumPy, Seaborn, Matplotlib

## Files
- `diabetes_analysis.py` – Main analysis script
- `diabetes_cleaned.csv` – Cleaned data
- `diabetes_visualizations.png` – Generated plots

## What I Learned (BME perspective)
Proper preprocessing of clinical data is critical — treating zero insulin as a real value would distort any downstream analysis or device algorithm.
