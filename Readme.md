# 📡 Telco Customer Churn Prediction

A machine learning project to predict whether a telecom customer will churn, built on a fictional dataset of 7,043 California customers from Q3.

**Dataset:** [Kaggle — Telco Customer Churn 11-1-3](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)

---

##  Problem Statement

Customer churn is one of the most costly problems in the telecom industry. This project builds a binary classification model to identify customers who are likely to leave — enabling the business to take proactive retention action before it's too late.

---

## Project Structure

```
telco-churn/
│
├── telco.ipynb          # Main notebook — full pipeline
├── telco.csv            # Dataset
└── README.md
```

---

## Pipeline Overview

```
Data Loading & Cleaning
        ↓
Exploratory Data Analysis
        ↓
Leakage Investigation (Satisfaction Score)
        ↓
Train / Validation / Test Split  (70 / 15 / 15)
        ↓
Scaling  (fit on train only)
        ↓
Baseline — Logistic Regression
        ↓
XGBoost  (GridSearchCV + Early Stopping)
        ↓
Model Comparison & Investigation
        ↓
Final Model Selection
```

---

## Dataset

| Property | Value |
|---|---|
| Rows | 7,043 |
| Columns | 50 (raw) |
| Target | `Churn Label` (Yes / No) |
| Churn Rate | ~26.5% |
| Class Ratio | ~2.8x more non-churners than churners |

---

## Data Cleaning

### Columns Dropped

| Column | Reason |
|---|---|
| `Churn Score` | Generated after churn — pure leakage |
| `Customer Status` | Directly encodes the target |
| `Churn Category` | Post-event, 70% missing |
| `Churn Reason` | Post-event, 70% missing |
| `Offer` | 55.5% missing — imputing would fabricate data |
| `Zip Code`, `Latitude`, `Longitude` | High-cardinality geographic noise |
| `Customer ID` | Pure ID column, no signal |
| Zero-variance columns | No predictive value |

### Missing Values

| Column | Action |
|---|---|
| `Internet Type` | Filled with `"No Internet"` — customers without internet service |

---

## Leakage Investigation — Satisfaction Score

`Satisfaction Score` was investigated separately because it correlated very strongly with churn. A quick test proved it was a **post-event proxy** — a survey filled out after the customer had already decided to leave:

| Model | AUC with Satisfaction Score | AUC without |
|---|---|---|
| Logistic Regression | ~1.00 | ~0.91 |

**Drop confirmed.** A ~0.09 AUC drop shows it was doing all the heavy lifting artificially. Removing it gives a model that would actually work in production, where satisfaction scores aren't available before churn.

---

## Exploratory Data Analysis

### Key Findings — Numeric Features

| Feature | Observation |
|---|---|
| Tenure | Churners have significantly lower tenure — newer customers churn more |
| Monthly Charge | Churners pay more on average |
| Total Charges | Lower for churners due to shorter tenure |
| Age | Older customers churn slightly more |

### Key Findings — Categorical Features

| Feature | Observation |
|---|---|
| Contract Type | Month-to-month customers churn at ~3× the rate of annual/2-year customers — **strongest predictor** |
| Dependents | Customers without dependents churn significantly more |
| Internet Type | Fiber optic customers churn at much higher rates |
| Add-ons (Security, Support, Backup) | Absence of each add-on correlates strongly with churn |
| Gender | No meaningful difference — contributes little |

---

## Modeling

### Train / Validation / Test Split

```
Total: 7,043 customers
├── Train : 70%  (~4,930 rows)  — model learns
├── Val   : 15%  (~1,056 rows)  — early stopping signal + threshold tuning
└── Test  : 15%  (~1,057 rows)  — reported ONCE at the very end
```

Stratified split used throughout to preserve the 73/27 class ratio in all sets.

### Scaling

`StandardScaler` is fitted **only on the training set**, then applied (transform only) to val and test — no data leakage.

---

### Model 1 — Logistic Regression (Baseline)

```python
LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
```

| Metric | Score |
|---|---|
| ROC-AUC (Val) | ~0.91 |
| ROC-AUC (Test) | ~0.91 |

`class_weight='balanced'` handles the 73/27 class imbalance automatically.

---

### Model 2 — XGBoost

```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=2.77,        # handles class imbalance
    early_stopping_rounds=100,    # prevents overfitting
    n_estimators=2000,            # high ceiling — ES stops early
)
```

**Hyperparameter tuning via GridSearchCV (5-fold CV):**

| Parameter | Search Space | Purpose |
|---|---|---|
| `n_estimators` | [300, 500] | Number of trees |
| `max_depth` | [3, 4, 5] | Tree complexity |
| `learning_rate` | [0.01, 0.05, 0.1] | Step size per tree |
| `subsample` | [0.8, 1.0] | Random row sampling — prevents overfitting |
| `colsample_bytree` | [0.8, 1.0] | Random feature sampling — prevents overfitting |
| `reg_alpha` | [0, 0.1, 1.0] | L1 regularization — removes weak features |
| `reg_lambda` | [0.5, 1.0, 2.0] | L2 regularization — shrinks all feature weights |

**Early stopping** monitors validation AUC after every boosting round and stops when it hasn't improved for 100 consecutive rounds — the model trained for **925 trees** before stopping.

| Metric | Score |
|---|---|
| ROC-AUC (Val) | ~0.91 |
| ROC-AUC (Test) | ~0.91 |

---

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | ~0.91 |
| XGBoost | ~0.91 |

### ROC-AUC of 0.91 — What it means

> If you randomly pick one real churner and one real non-churner from the dataset, the model will correctly rank the churner as more likely to churn **91% of the time.**

| AUC Range | Interpretation |
|---|---|
| 0.50 | Random guessing |
| 0.70–0.80 | Acceptable |
| 0.80–0.90 | Good |
| **0.91** | **Strong** |
| 0.95+ | Excellent |

---

## Why Do LR and XGBoost Perform the Same?

Three metrics were investigated to understand why both models achieved identical AUC:

| Metric | Value | Interpretation |
|---|---|---|
| Probability correlation | 0.957 | High but not identical — models learned **similar but different** patterns |
| Trees used by XGBoost | 925 | XGBoost worked hard — this is **not** a lazy or undertrained model |
| Top 3 feature importance | 40.5% | Importance spread across many features — **no single feature dominates** |

**Conclusion:** The data is not simply linear. Both models hit the same **performance ceiling of ~0.91** that exists in this dataset after removing Satisfaction Score. The remaining behavioral features collectively cap predictive power at ~0.91 — both models reach that ceiling through different learned patterns.

---

##  Final Model — Logistic Regression

Logistic Regression was selected as the final model — not because the data is linear, but because it **matches XGBoost's performance** with far less complexity:

- Faster to train and deploy
- Cheaper to run in production
- Coefficients are directly interpretable by business stakeholders
- No hyperparameter tuning overhead

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
xgboost
kagglehub
```

---

## How to Run

1. Clone the repo
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn xgboost kagglehub`
3. Open `telco.ipynb`
4. Run all cells top to bottom

