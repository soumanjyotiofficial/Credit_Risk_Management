# German Credit Risk Modeling — Probability of Default (PD) Model

## Project Overview

This project focuses on building a **Probability of Default (PD) Model** using the German Credit Risk dataset.

The objective of the project is to estimate the probability that a borrower may default on a loan obligation using machine learning techniques.

The project follows a complete credit risk modeling workflow including:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- Random Forest Classification
- PD Estimation
- ROC-AUC Evaluation
- Gini Coefficient Analysis

---

# What is Probability of Default (PD)?

Probability of Default (PD) represents the likelihood that a borrower will fail to meet debt obligations within a specified time horizon.

PD is one of the most important concepts in:
- Credit Risk Analytics
- Basel Framework
- Risk Management
- Banking Analytics
- IFRS 9 Modeling

Mathematically:

```math
0 \leq PD \leq 1
```

Where:
- `PD = 0` → No default risk
- `PD = 1` → Certain default

In this project:
- `good borrower = 0`
- `bad borrower = 1`

The machine learning model estimates the probability that a borrower belongs to the default class.

---

# Dataset

Dataset Used:
- German Credit Risk Dataset

Target Variable:
- `Risk`

Target Mapping:

```python
df['target'] = df['Risk'].map({'bad': 1, 'good': 0})
```

Where:
- `1 = Default`
- `0 = Non-Default`

---

# Business Objective

The primary objective is to build a PD model that helps financial institutions:

- identify high-risk borrowers
- reduce credit losses
- improve underwriting decisions
- estimate expected loss
- support Basel capital requirements

---

# Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-Learn

---

# Workflow

## 1. Data Collection

```python
df = pd.read_csv("/content/german_credit_data.csv")
```

---

## 2. Target Variable Creation

```python
df['target'] = df['Risk'].map({'bad': 1, 'good': 0})
```

The target variable represents borrower default status.

---

## 3. Data Cleaning

Missing values were removed.

```python
df = df.dropna().reset_index(drop=True)
```

Unused columns were dropped.

```python
df.drop(columns='Unnamed: 0', inplace=True)
```

---

## 4. Exploratory Data Analysis (EDA)

EDA was performed using:
- Boxplots
- Descriptive Statistics
- Outlier Analysis

Variables analyzed:
- Duration
- Credit Amount
- Age

The analysis helped understand:
- borrower behavior
- loan distribution
- exposure concentration
- potential outliers

---

## 5. Feature Engineering

Categorical variables were transformed using One-Hot Encoding.

Encoded Features:
- Sex
- Job
- Housing
- Saving accounts
- Checking account
- Purpose

The optimized model excluded:
- Purpose

to improve model performance.

---

## 6. PD Model Development

A Random Forest Classifier was used for PD estimation.

```python
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
```

The dataset was split into:
- Training Set
- Testing Set

```python
train_test_split(test_size=0.2)
```

The model estimates the probability that a borrower defaults.

---

# PD Model Evaluation

The model was evaluated using:

## ROC-AUC Score

ROC-AUC measures the model's ability to distinguish between:
- default borrowers
- non-default borrowers

Higher ROC-AUC indicates stronger discriminatory power.

---

## Gini Coefficient

The Gini coefficient is a widely used metric in:
- banking
- credit scoring
- Basel risk models

Formula:

```math
Gini = (2 \times ROC\text{-}AUC) - 1
```

Higher Gini indicates better credit risk separation.

---

# Initial PD Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.5951 |
| Gini Coefficient | 0.1902 |

---

# Optimized PD Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.6869 |
| Gini Coefficient | 0.3738 |

---

# Model Improvement

The optimized PD model achieved better performance through:
- improved feature selection
- removal of less informative variables
- improved encoding strategy
- better handling of categorical variables

The optimized model demonstrated:
- improved ROC-AUC
- higher Gini coefficient
- stronger borrower discrimination

---

# Credit Risk Interpretation

A borrower with:
- higher predicted probability → higher PD
- lower predicted probability → lower PD

The model helps identify:
- risky customers
- high default segments
- portfolio vulnerabilities

---

# Expected Loss Framework

The PD model can be integrated into Expected Loss estimation.

```math
EL = PD \times LGD \times EAD
```

Where:
- `PD` = Probability of Default
- `LGD` = Loss Given Default
- `EAD` = Exposure At Default

---

# Project Structure

```text
├── german_credit_data.csv
├── credit_risk_model.ipynb
├── README.md
```

---

# Future Improvements

Possible enhancements:
- Logistic Regression Scorecard
- WoE Encoding
- Information Value (IV)
- Hyperparameter Tuning
- XGBoost
- Cross Validation
- SHAP Explainability
- Probability Calibration
- Basel IRB Framework
- Feature Importance Analysis

---

# Conclusion

This project demonstrates an end-to-end implementation of a Probability of Default (PD) model using machine learning techniques on the German Credit Risk dataset.

The project provides practical exposure to:
- credit risk analytics
- PD modeling
- machine learning in finance
- risk evaluation metrics
- financial data preprocessing

and demonstrates how machine learning can support modern credit risk assessment frameworks.

---

# Author

Souman Jyoti

---

# License

This project is for educational and research purposes.
