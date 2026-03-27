# Pima Indian Diabetes Risk Analysis

**Biomedical Engineering Portfolio Project**

This project demonstrates key data preprocessing and exploratory analysis skills using the classic **Pima Indians Diabetes Dataset** — commonly used in biomedical and clinical research.

## Project Goals
- Properly handle medically impossible zero values in clinical measurements (Glucose, BMI, Insulin, etc.)
- Perform feature engineering by creating clinically relevant **BMI categories**
- Explore relationships between key risk factors and diabetes outcome
- Create publication-style visualizations

## Key Insights
- 34.9% diabetes prevalence in the dataset
- Significant missing/impossible values: 374 Insulin records and 227 SkinThickness records were invalid zeros
- Obese patients (BMI ≥ 30) form the largest group and show higher diabetes rates
- Strong visual correlation between elevated Glucose and diabetes outcome

## Technologies Used
- Python 3
- pandas, NumPy
- seaborn & matplotlib (for clinical visualizations)

## Files
- `diabetes_analysis.py` — Main analysis script
- `diabetes.csv` — Original dataset
- `diabetes_cleaned.csv` — Cleaned version with NaNs and BMI categories
- `diabetes_visualizations.png` — Generated plots

## BME Relevance
In biomedical engineering, accurate data cleaning is critical before developing diagnostic algorithms, wearable devices, or predictive models. Treating zero insulin as a valid measurement would lead to incorrect conclusions in any real clinical or device application.

---

**Author:** Michelle Longog  
**Date:** March 2026