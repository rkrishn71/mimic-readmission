# 30-Day Hospital Readmission Prediction
### MIMIC-IV Clinical ML Project

A machine learning pipeline to predict 30-day hospital readmissions using 
electronic health record data from MIMIC-IV. Built as a portfolio project 
demonstrating end-to-end clinical ML: cohort construction, feature engineering, 
modeling, explainability, and fairness analysis.

---

## Results Summary

| Model | Test AUC-ROC | Test AP Score |
|-------|-------------|---------------|
| Logistic Regression (baseline) | 0.6317 | 0.2651 |
| XGBoost | **0.6700** | **0.3074** |

- XGBoost achieves **6.1% improvement** over logistic regression baseline
- AP Score of 0.3074 is **1.77x above random baseline** (0.174)
- Flagging top 10% highest-risk patients catches **20% of readmissions** at 2.2x baseline precision

---

## Research Credentialing

Access to MIMIC-IV required completing the full PhysioNet credentialing 
process, including:

- **CITI Program certification** in Data or Specimens Only Research, 
  covering human subjects protection, HIPAA privacy regulations, research 
  ethics (Belmont Report principles), conflict of interest, and genetic 
  research considerations
- **PhysioNet credentialing** with institutional affiliation verification
- **MIMIC-IV Data Use Agreement** — binding legal agreement governing 
  data handling, storage, and publication

This credentialing process mirrors the requirements for working with 
protected health information (PHI) in clinical research and industry 
settings, including familiarity with de-identification standards, 
minimum necessary principles, and responsible data handling practices.

## Clinical Context

30-day readmission is a key quality metric in US healthcare. The Centers for 
Medicare and Medicaid Services (CMS) penalizes hospitals for excess readmissions 
under the Hospital Readmissions Reduction Program, affecting reimbursement for 
millions of patients annually.

This project follows the **CMS definition**: any unplanned readmission within 
30 days of discharge, regardless of diagnosis. The target use case is **risk 
stratification at discharge** — identifying high-risk patients for targeted 
interventions such as follow-up calls, medication reconciliation, or home health 
visits.

---

## Dataset

**Source:** [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) — 
de-identified EHR data from Beth Israel Deaconess Medical Center, Boston MA.
Access requires PhysioNet credentialing and CITI training completion.

| Table | Rows | Description |
|-------|------|-------------|
| admissions | 546,028 | Hospital admission records |
| patients | 364,627 | Patient demographics |
| diagnoses_icd | 6,364,488 | ICD-9/10 diagnosis codes |
| prescriptions | 20,292,611 | Medication orders |

**Final cohort:** 534,152 adult admissions after excluding in-hospital deaths
and data quality issues. **Readmission rate: 17.4%**

---

## Project Structure

mimic-readmission/
├── notebooks/
│   ├── 01_data_extraction.ipynb      # Cohort construction, target variable
│   ├── 02_feature_engineering.ipynb  # Feature engineering from 4 tables
│   ├── 03_modeling.ipynb             # Logistic regression + XGBoost
│   └── 04_evaluation.ipynb           # Calibration, SHAP, fairness analysis
├── src/
│   ├── features.py                   # Reusable feature engineering functions
│   └── model.py                      # Model training utilities
├── figures/                          # All generated plots
└── README.md

**Note:** The `data/` directory is excluded from this repository per the 
MIMIC-IV data use agreement. Access requires independent PhysioNet credentialing.

---

## Methodology

### Cohort Construction (Notebook 1)

- Excluded in-hospital deaths (11,801 admissions) — these patients cannot 
  be readmitted and represent a distinct outcome
- Excluded same-day readmissions < 24 hours (13,609 admissions) — likely 
  transfers rather than true readmissions
- Removed data quality issues: negative LOS (74) and extreme LOS > 365 days (1)
- **Temporal train/test split** — admissions before cutoff date for training, 
  after for testing. This prevents data leakage from future information 
  contaminating training, which is critical in time-series clinical data

### Feature Engineering (Notebook 2)

45 features engineered across 4 groups:

**Demographic** (2 features)
- Age at admission (computed from MIMIC anchor_age/anchor_year de-identification scheme)
- Gender

**Admission** (27 features)
- Length of stay, admission hour, day of week, weekend flag
- Admission type (8 categories one-hot encoded)
- Discharge location (12 categories) — clinically strong predictor
- Insurance type (4 categories) — proxy for socioeconomic status
- Marital status (3 categories) — proxy for social support

**Comorbidity** (7 features from diagnoses_icd)
- Total diagnosis count — proxy for medical complexity
- Binary flags for 6 high-risk conditions: heart failure, renal failure, 
  diabetes, COPD, pneumonia, sepsis — mapped from both ICD-9 and ICD-10 codes

**Medication** (5 features from prescriptions)
- Total unique medication count — polypharmacy is a known readmission risk factor
- Binary flags for 4 high-risk drug classes: insulin, anticoagulants, 
  opioids, diuretics

### Modeling (Notebook 3)

**Class imbalance handling:** `scale_pos_weight = 4.82` 
(ratio of negative to positive training examples)

**XGBoost configuration:**
```python
xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4.82,
    early_stopping_rounds=20
)
```

Early stopping prevented overfitting — training halted when validation AUC 
stopped improving. Final train/test AUC gap of 0.043 indicates mild, 
acceptable overfitting.

---

## Key Findings

### Feature Importance (SHAP)

Top predictors by mean absolute SHAP value:

1. **n_medications** — medication count is the single strongest predictor, 
   reflecting polypharmacy complexity
2. **n_diagnoses** — overall comorbidity burden
3. **age_at_admission** — non-linear effect not captured by simple correlation
4. **discharge_location** — particularly hospice (low risk) and home health care (high risk)
5. **admission_type** — direct emergency admissions carry highest risk

### Notable Non-Linear Finding: Length of Stay

SHAP analysis revealed a **U-shaped relationship** between LOS and readmission 
risk — both very short stays (premature discharge before stabilization) and very 
long stays (high-acuity illness) predict readmission. This non-linearity is 
captured by XGBoost but would be missed by logistic regression, partially 
explaining the performance gap between models.

### Individual Patient Explanations

SHAP waterfall plots provide clinically coherent individual explanations:

- **High-risk patient:** 77yo, 42 medications, 19 diagnoses, renal failure, 
  discharged to home health care → predicted high readmission probability
- **Low-risk patient:** younger, 5 medications, 5 diagnoses, private insurance, 
  discharged to psychiatric facility → predicted low readmission probability

Feature directions in all cases align with established clinical knowledge, 
validating that the model learned genuine clinical patterns rather than 
spurious correlations.

### Clinical Utility

| Intervention Threshold | Patients Flagged | Readmissions Caught | Recall | Precision |
|----------------------|-----------------|--------------------| -------|-----------|
| Top 5% | 5,341 | 2,169 | 11.2% | 40.6% |
| Top 10% | 10,683 | 3,827 | 19.8% | 35.8% |
| Top 15% | 16,024 | 5,381 | 27.8% | 33.6% |
| Top 20% | 21,366 | 6,724 | 34.8% | 31.5% |
| Top 30% | 32,049 | 9,294 | 48.1% | 29.0% |

**Practical interpretation:** A care coordinator following up with the top 10% 
highest-risk patients will reach a readmitted patient in 1 of every 2.8 calls, 
compared to 1 in 5.5 with random selection — more than doubling intervention yield.

---

## Fairness Analysis

The model shows **meaningful performance disparities** across patient subgroups:

| Subgroup | N | Readmission Rate | AUC |
|----------|---|-----------------|-----|
| Private insurance | 31,785 | 15.3% | **0.720** |
| Other insurance | 2,418 | 18.8% | 0.705 |
| Medicaid | 20,736 | 19.8% | 0.655 |
| Medicare | 50,424 | 19.3% | 0.634 |

| Age Group | N | Readmission Rate | AUC |
|-----------|---|-----------------|-----|
| 18-40 | 18,561 | 14.1% | **0.721** |
| 41-60 | 32,001 | 19.7% | 0.671 |
| 61-75 | 31,197 | 19.6% | 0.662 |
| 75+ | 24,877 | 17.3% | 0.615 |

**The model performs worst on Medicare and elderly patients — the exact 
population where readmission prediction matters most.** The AUC gap between 
Private and Medicare patients (0.086) and between 18-40 and 75+ patients 
(0.106) likely reflects that older, multimorbid patients have complex 
presentations driven by subtle physiological changes (fluid balance, renal 
function decline, medication interactions) that are captured in lab values 
and vital signs but not in the administrative features used here.

**Clinical deployment recommendation:** This model should not be used for 
resource allocation decisions without subgroup-specific validation, particularly 
for Medicare and elderly patients. Incorporating lab values and vital signs is 
expected to reduce these performance gaps.

---

## Limitations

- **Administrative features only** — no lab values, vital signs, or clinical 
  notes. Published models using full MIMIC feature sets report AUC 0.72-0.78
- **Single academic medical center** — BIDMC serves a specific patient 
  population; performance may not generalize to community hospitals or 
  safety-net institutions
- **Planned readmissions not excluded** — identifying planned readmissions 
  from administrative data requires additional logic not implemented here
- **All-cause readmission definition** — includes readmissions unrelated to 
  the index admission (e.g. trauma), consistent with CMS definition but 
  potentially noisy
- **Subgroup performance gaps** — require mitigation before clinical deployment

---

## Reproducing This Project

### Prerequisites

- Python 3.10
- MIMIC-IV access via PhysioNet (requires credentialing)
- Conda

### Setup

```bash
git clone https://github.com/rkrishn71/mimic-readmission.git
cd mimic-readmission
conda create -n mimic-readmission python=3.10
conda activate mimic-readmission
pip install pandas numpy scikit-learn xgboost==1.7.6 shap matplotlib seaborn jupyter
```

### Data

Download the following tables from MIMIC-IV v3.1 (requires PhysioNet access):

```bash
wget --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz

wget --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz

wget --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz

wget --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz
```

Place all files in the `data/` directory. Then run notebooks 1-4 in order.

---

## Future Work

- **Richer features:** Lab values (creatinine, hemoglobin, BMP) and vital 
  signs from MIMIC ICU module expected to improve AUC toward 0.72-0.78 and 
  reduce subgroup performance gaps
- **Elixhauser Comorbidity Index:** Replace individual condition flags with 
  the validated 31-category Elixhauser index using the `pyaomop` library
- **Diagnosis-specific models:** Separate models for heart failure, COPD, 
  and pneumonia — the three CMS-tracked conditions — may outperform a 
  single all-cause model
- **Calibration:** Apply Platt scaling or isotonic regression to improve 
  probability calibration, critical for clinical decision support
- **Natural language processing:** Discharge summaries from MIMIC-IV-Note 
  contain rich clinical context not captured in structured data

---

## References

- Johnson, A. et al. (2023). MIMIC-IV (version 3.1). PhysioNet.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Lundberg, S. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Elixhauser, A. et al. (1998). Comorbidity Measures for Use with Administrative Data. Medical Care.
- Kansagara, D. et al. (2011). Risk Prediction Models for Hospital Readmission. JAMA.

---

## Data Access

This project uses MIMIC-IV, a restricted dataset requiring:
1. Completion of CITI "Data or Specimens Only Research" training
2. PhysioNet credentialing
3. Signing the MIMIC-IV data use agreement

See [PhysioNet](https://physionet.org/content/mimiciv/3.1/) for access instructions.

---

*Built with MIMIC-IV | XGBoost | SHAP | scikit-learn*