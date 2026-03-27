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
- Calculated **Length of Stay** from admission/discharge dates
- Performed detailed aggregation by Medical Condition (LOS, billing, age)
- Added data quality checks and summary statistics
- Conducted one-way ANOVA test on Length of Stay across conditions (p = 0.324 — no significant differences)
- Created enhanced visualizations including correlation heatmap and test result distributions
- Explored patterns in Emergency vs Elective admissions and age distributions

**Key Findings**:
- Average hospital stay: **15.51 days** across all conditions
- Very similar length of stay and billing amounts across Diabetes, Hypertension, Obesity, Cancer, Asthma, and Arthritis
- Balanced test results (~33% Abnormal / Inconclusive / Normal)
- No statistically significant difference in LOS by medical condition (ANOVA)

**Technologies Used**: Python, pandas, seaborn, matplotlib, scipy

## Repository Structure
- `diabetes_analysis.py` → Diabetes prediction project
- `healthcare_analysis.py` → Hospital admissions analysis (enhanced)
- Visualizations and cleaned datasets for both projects

---

This repository demonstrates core BME skills: clinical data preprocessing, feature engineering, exploratory analysis, statistical testing, and basic predictive modeling.

**Last updated:** March 2026