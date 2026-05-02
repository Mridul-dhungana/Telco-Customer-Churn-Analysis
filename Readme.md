<img width="200" height="200" alt="UTA-DataScience-Logo" src="https://github.com/user-attachments/assets/9431c15f-2f1a-4422-9857-4cf2deb36b8e" />


# 📡 Telco Customer Churn Prediction

A machine learning project to predict whether a telecom customer will churn, built on a fictional dataset of 7,043 California customers from Q3.

**Dataset:** [Kaggle — Telco Customer Churn 11-1-3](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)

---
## Overview
The project started with a simple goal to predict which telecom customers will churn.
**The Trap** 
Early on, Logistic Regression hit a ROC-AUC of ~1.00 on the validation set. Seeing a model learn something with almost complete accuracy made the things suspicious. Early on, on the numeric visulization part we had figured out satisfaction score being the best predictor of the churn. And training the data with the satisfaction score proved it to be true. 

But, we wanted to make it more resilient, drop the satisfaction score, even if it was not mentioned to be a leakage column in the original dataset, and what I got after it was a, the AUC dropped to 0.91, which was much more reasonable to me.

**Interesting part** <br>
The interesting part was when I decided to use XG Boost model, hypertune it with GRIDSearch CV to train the model, my ROC-AUC score got capped in 0.91, **the exact same as the simple basline. I got interested in it and first checked, whether the data was too linear?
**Probability Correlation** of **0.957** showed model learned similar but different pattern, **925 trees used by XGBoost** showed that the model was not understrained and **40.5%** importance of top 3 features show that no single feature dominates, showing that it was not data being linear, but it was rather a performance ceiling with the remaining feature after removing **Satisfaction Score**

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
└──Other pictures
```

---

## Pipeline Overview

```
1. Load the Data, explore it perform data cleaning
        
2. Exploratory Data Analysis
        
3. Leakage Investigation (Satisfaction Score)
        
4. Train / Validation / Test Split  (70 / 15 / 15)
        
5. Scaling  (fit on train only)
        
6. Baseline — Logistic Regression
        
7. XGBoost  (GridSearchCV + Early Stopping)
        
8. Model Comparison & Investigation
        
9. Final Model Selection
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
<img width="923" height="390" alt="churn" src="https://github.com/user-attachments/assets/ec2f317a-8001-4f27-b7bf-dd50e673e85f" />


## Data Cleaning

### Columns Dropped

| Column | Reason |
|---|---|
| `Churn Score` | Generated after churn - pure leakage |
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

`Satisfaction Score` was investigated separately because it correlated very strongly with churn. Creating a model with it showed that the model became too accurate and the goal of predicting churn without even knowing the satisfaction score was not achieved. 

| Model | AUC with Satisfaction Score | AUC without |
|---|---|---|
| Logistic Regression | ~1.00 | ~0.91 |

**Drop confirmed** A ~0.09 AUC drop shows it was doing all the heavy lifting artificially. Removing it gives a model that would actually work in production, where satisfaction scores aren't available before churn.

<img width="1389" height="390" alt="Satisfaction" src="https://github.com/user-attachments/assets/5a606290-c55c-43a7-b1d4-061f3cb75b9d" />
<img width="585" height="455" alt="LR with satisfaction" src="https://github.com/user-attachments/assets/5c444373-f2a4-4f30-af6b-7cc3c91b479b" />


---

## Exploratory Data Analysis

### Key Findings — Numeric Features

| Feature | Observation |
|---|---|
| Tenure | Churners have significantly lower tenure; newer customers churn more |
| Monthly Charge | Churners pay more on average |
| Total Charges | Lower for churners due to shorter tenure |
| Age | Older customers churn slightly more |

<img width="1389" height="390" alt="Tenure" src="https://github.com/user-attachments/assets/4a1ef5d2-7c75-45cb-9d23-cc8b31453772" />

### Key Findings — Categorical Features

| Feature | Observation |
|---|---|
| Contract Type | Month-to-month customers churn at ~3× the rate of annual/2-year customers — **strongest predictor** |
| Dependents | Customers without dependents churn significantly more |
| Internet Type | Fiber optic customers churn at much higher rates |
| Add-ons (Security, Support, Backup) | Absence of each add-on correlates strongly with churn |
| Gender | No meaningful difference — contributes little |

---
<img width="1507" height="408" alt="Contract" src="https://github.com/user-attachments/assets/0a7cd071-b4db-405b-980f-def0c5c3d5bd" />

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

`StandardScaler` is fitted **only on the training set**, then applied (transform only) to val and test; this ensures there is no data leakage.

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
| Probability correlation | 0.957 | High but not identical; this means models learned **similar but different** patterns |
| Trees used by XGBoost | 925 | XGBoost worked hard, this is **not** a lazy or undertrained model |
| Top 3 feature importance | 40.5% | Importance spread across many features, **no single feature dominates** |

**Conclusion:** 
The data doesnot seem to be linear, rather both the models hit the same **performance ceiling of ~0.91** that exists in this dataset after removing Satisfaction Score. The remaining behavioral features collectively cap predictive power at ~0.91.

---

##  Final Model — Logistic Regression
<img width="789" height="590" alt="ROC_AUC" src="https://github.com/user-attachments/assets/3a1c09e3-8780-4797-9ef8-9e118dccfe4b" />
Logistic Regression was selected as the final model. It was not because the data was linear  but rather because it **matches XGBoost's performance** with far less complexity. It is 
faster to train, deploy, cheaper to run and requires no hyperparameter tuning.

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

