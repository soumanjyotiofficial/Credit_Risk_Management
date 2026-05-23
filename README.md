# Credit_Risk_Management

# German Credit Risk Modeling using Machine Learning

## Project Overview

This project focuses on building and optimizing a Probability of Default (PD) model using the German Credit Risk dataset.

The objective is to classify borrowers into:
- Good Credit Risk
- Bad Credit Risk

using machine learning techniques and evaluate the model using:
- ROC-AUC Score
- Gini Coefficient

The project demonstrates a complete credit risk modeling workflow including:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- Random Forest Classification
- Model Optimization
- Credit Risk Evaluation Metrics

---

# Dataset

Dataset Used:
- German Credit Risk Dataset

Target Variable:
- `Risk`

Mapping:
- `good → 0`
- `bad → 1`

---

# Problem Statement

Financial institutions need to estimate the probability that a borrower may default on a loan.

This project aims to:
- predict borrower default risk
- identify risky customers
- improve model discriminatory power
- evaluate credit risk using industry-standard metrics

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

The dataset was loaded using pandas.

```python
df = pd.read_csv("/content/german_credit_data.csv")
```

---

## 2. Target Variable Creation

```python
df['target'] = df['Risk'].map({'bad': 1, 'good': 0})
```

Where:
- `1 = Default`
- `0 = Non-Default`

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

## 4. Exploratory Data Analysis

Boxplots were used to analyze:
- Duration
- Credit Amount
- Age

This helped identify:
- outliers
- skewness
- exposure distribution

---

## 5. Feature Engineering

Categorical variables were encoded using One-Hot Encoding.

Features encoded:
- Sex
- Job
- Housing
- Saving accounts
- Checking account
- Purpose

Optimized model excluded:
- Purpose

---

## 6. Model Building

A Random Forest Classifier was used.

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

---

# Model Evaluation Metrics

The model was evaluated using:

## ROC-AUC Score

Measures the model's ability to distinguish between:
- good borrowers
- bad borrowers

Higher ROC-AUC indicates better discrimination.

---

## Gini Coefficient

Widely used in:
- Banking
- Basel Framework
- Credit Risk Analytics

Formula:

```math
Gini = (2 × ROC-AUC) - 1
```

---

# Initial Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.5951 |
| Gini Coefficient | 0.1902 |

---

# Optimized Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.6869 |
| Gini Coefficient | 0.3738 |

---

# Improvement Achieved

The optimized model improved predictive performance by:
- removing less informative variables
- improving feature selection
- handling categorical variables more effectively

The optimized model achieved:
- higher ROC-AUC
- higher Gini score
- better discriminatory power

---

# Business Interpretation

A higher Gini coefficient indicates:
- stronger risk separation
- better borrower classification
- improved credit underwriting capability

The optimized model demonstrates improved ability to:
- identify high-risk borrowers
- estimate default probability
- support credit decision-making

---

# Credit Risk Concepts

## Probability of Default (PD)

Probability that a borrower defaults on obligations.

---

## Exposure at Default (EAD)

Total exposure at the time of borrower default.

---

## Loss Given Default (LGD)

Percentage loss incurred after borrower default.

---

## Expected Loss Formula

```math
EL = PD \times LGD \times EAD
```

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
- XGBoost
- Hyperparameter Tuning
- SHAP Explainability
- Feature Importance Analysis
- Cross Validation
- Probability Calibration
- Basel PD Modeling Framework
- WoE Encoding
- IV Analysis

---

# Conclusion

This project demonstrates an end-to-end implementation of a machine learning-based credit risk model using the German Credit dataset.

The project covers:
- data preprocessing
- feature engineering
- risk modeling
- performance evaluation
- business interpretation

and provides a practical introduction to:
- Probability of Default Modeling
- Credit Risk Analytics
- Financial Machine Learning

---

# Author

Souman Jyoti

---

# License

This project is for educational and research purposes.
