# Healthcare & Biomedical Data Analysis Portfolio

**Michael Lozada-Longog**  
Biomedical Engineering Projects

## Project 1: Pima Indian Diabetes Risk Analysis
- Cleaned clinical data by handling medically impossible zero values  
- Feature engineering: Created BMI categories (Underweight, Normal, Overweight, Obese)  
- Built Logistic Regression model → **75.3% accuracy**  
- Visualized relationships between Glucose, Insulin, BMI and diabetes outcome  

**Key BME Insight**: Proper preprocessing is critical for clinical algorithms and diagnostic tools.

## Project 2: Healthcare Admissions Analysis (55,500 patients)
- Calculated Length of Stay from admission/discharge dates  
- Detailed aggregation by Medical Condition (LOS, billing, age)  
- Added data quality checks, one-way ANOVA test (p = 0.324 — no significant differences in LOS)  
- Enhanced visualizations including correlation heatmap and test result distributions  

**Key Findings**:  
- Average hospital stay: **15.51 days** across all conditions  
- Very similar LOS and billing amounts across Diabetes, Hypertension, Obesity, Cancer, Asthma, Arthritis  
- Balanced test results (~33% each category)  

**Technologies**: Python, pandas, seaborn, matplotlib, scipy

## Project 3: Heart Disease UCI Risk Prediction (920 patients)
- Binary classification (presence of heart disease)  
- Handled missing clinical values with median imputation  
- Logistic Regression + Random Forest models  
- Added feature importance analysis (Random Forest)  
- EDA with correlation heatmap, age/cholesterol/BP distributions, and chest pain type  

**Key BME Insight**: Chest pain type, age, and cholesterol are among the strongest predictors — directly relevant to diagnostic tool development.

**Technologies**: Python, pandas, seaborn, matplotlib, scikit-learn

## Repository Structure
- `diabetes_analysis.py` → Diabetes prediction project  
- `healthcare_analysis.py` → Hospital admissions analysis  
- `heart_disease_analysis.py` → Heart disease risk prediction  
- Visualizations + cleaned datasets for all three projects  

---

This repository demonstrates core Biomedical Engineering skills:  
clinical data preprocessing, feature engineering, exploratory analysis, statistical testing, and predictive modeling for real-world healthcare applications.

**Last updated:** March 2026