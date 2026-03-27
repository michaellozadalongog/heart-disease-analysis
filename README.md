# Pima Indian Diabetes Risk Analysis

**Biomedical Engineering Portfolio Project**

Professional exploratory data analysis and basic predictive modeling on the classic Pima Indians Diabetes Dataset.

## Key Features
- Handled medically impossible zero values (Glucose = 0, Insulin = 0, etc.) by converting to NaN
- Imputed missing values using median (appropriate for clinical data)
- Feature engineering: Created clinical **BMI categories** (Underweight, Normal, Overweight, Obese)
- Visualized relationships using scatter plots, boxplots, and correlation heatmap
- Built a **Logistic Regression model** to predict diabetes outcome

## Results
- Diabetes prevalence: 34.9%
- Logistic Regression Accuracy: **75.3%**
- Strongest correlations: Glucose and BMI with Outcome

## Technologies
- Python, pandas, NumPy, seaborn, matplotlib
- scikit-learn (Logistic Regression + evaluation)

## Files
- `diabetes_analysis.py` — Complete analysis and modeling script
- `diabetes_cleaned.csv` — Cleaned and imputed dataset
- `diabetes_visualizations.png` — Generated plots (including correlation heatmap)

## BME Relevance
In biomedical engineering, proper data preprocessing is essential before developing diagnostic tools, wearable glucose monitors, or AI-based clinical decision support systems. This project demonstrates awareness of clinical data quality issues and basic predictive modeling skills.

---

**Author:** Michelle Longog  
**Date:** March 2026