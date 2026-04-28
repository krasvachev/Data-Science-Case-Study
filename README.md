# LittleBank — Data Science Case Study

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931a?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-006400)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![Last Commit](https://img.shields.io/github/last-commit/krasvachev/Data-Science-Case-Study)](https://github.com/krasvachev/Data-Science-Case-Study)
[![Stars](https://img.shields.io/github/stars/krasvachev/Data-Science-Case-Study?style=social)](https://github.com/krasvachev/Data-Science-Case-Study)

> **A complete, interview-ready Data Science Case Study for Big Four (Deloitte, EY, KPMG, PwC) technical interviews — featuring an exhaustive EDA, a full ML pipeline, business-focused recommendations and a detailed interview preparation guide.**

---

## Introduction

A Data Scientist 📊 or Machine Learning Engineer 📈 role at one of the Big Four companies (Deloitte, EY, KPMG, PwC)... That's a great opportunity that should not be missed. The Big Four companies generate  $219 billion in revenue 💵, with worldwide offices and more than 1.5 million employees 🧑‍💼. This makes them the largest professional service and accounting companies in the world 🧮. Being a part of them means you could work on a variety of projects across a wide range of fields and build broad experience across many topics — especially in the early stages of your career 🚀. However, to be a part of the Big Four, first you have to pass the interview process. And the case study task... I mean the technical case study 👨‍💻.

What is a case study 😦🤨? That is the question I asked myself when I first heard that I had to tackle such a problem. A coding task or implementing an ML algorithm, we all get that. But what a case study actially is 🤷‍♂️🤷‍♂️🤷‍♂️? Typically, there are between 3 and 5 steps when applying to a Big Four company. The case study is a business task that the candidate has to solve within a fixed time ⌛. For the tech interview, the company often provides a real-world dataset. The goal is to perform Exploratory Data Analysis (EDA) 📊, apply Machine Learning (ML) models 📈 and answer specific business questions connected to the task 📝. Usually, the time to solve all three tasks is 3 hours.

When I was doing my preparation, I could not find a case study example for Data Scientists or ML Engineers. This is the first reason for creating this repository — to provide job seekers with an exercise to practise on 🔍. There is also a second reason: I failed to pass the data science case study. However, I decided to solve the task outstandingly with a solution that successfully passes the case study interview. And last but not least, the goal of the repo is not just helping me, but also helping **you** land that dream job 💪💪💪.

The repo provides an exhaustive solution to a Data Science case study task given during a Big Four technical interview. It also provides guidance on how to prepare for the interviews 📋. The main focus is put on how to tackle the case study. There are also hints on how to use LLM models to help you efficiently during your interview preparation 🤖.

The solutions are not limited to the Big Four accounting companies. They can be helpful for other accounting and professional service firms as well 🎯🎯🎯. In addition, the solutions apply the most common ML and Data Science practices. That is why they can be a great resource for any ML Engineer or Data Scientist preparing for a tech interview 🎯🎯🎯.

---

## Table of Contents

1. [Case Study Instructions](#case-study-instructions)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Brief and Detailed Solutions](#brief-and-detailed-solutions)
5. [Structure of the Notebooks](#structure-of-the-notebooks)
6. [Solutions](#solutions)
   - [Task 1 — Exploratory Data Analysis](#task-1--exploratory-data-analysis-eda)
   - [Task 2 — Machine Learning](#task-2--machine-learning)
7. [Models and Accuracy](#models-and-accuracy)
8. [Conclusion and Key Insights](#conclusion-and-key-insights)
9. [How to Tackle the Interview](#how-to-tackle-the-interview)
10. [Python Scripts](#python-scripts)
11. [Requirements](#requirements)
12. [Contributing](#contributing)
13. [License](#license)

---

## Case Study Instructions

### Customer Analytics Case Study — Cross-sell Opportunities for LittleBank

<mark>**Client:**</mark> LittleBank — a retail bank providing deposit accounts, loans and savings products.

<mark>**Problem:**</mark> The head of loan sales has noticed a recent **drop in subscriptions of the "classic savings account"** product, despite consistent telemarketing efforts offering the account to customers. He has turned to our company for advice on how to improve sales of this product.

<mark>**Data:**</mark> LittleBank has shared a [data file](https://github.com/krasvachev/Data-Science-Case-Study/tree/749f388c5ddd75bffebf4815294aec153770be96/data) (`LittleBank_Case_Study.csv`) containing historical telemarketing-campaign records. 

The file includes:

- (i) Attributes of customer contacts when a classic savings product was offered.
- (ii) Details of any previous campaigns where a similar product had been offered.
- (iii) Customer attributes.
- (iv) Macroeconomic and environmental factors at the time each contact was made.
- (v) An indicator variable showing whether or not the client bought the product.

Since LittleBank has not yet used advanced analytics in its sales and marketing activities, the candidate must come prepared to **describe every algorithm employed and the approach taken** in plain, non-technical language. To conduct your analysis, please develop a Python Jupyter notebook for the coding part of the questions below. 

### Business Questions

| # | Task |  Type  |
|---|------|:------:|
| **1** | What steps would you take to **understand and clean this data**? Perform **Exploratory Data Analysis (EDA)**. | Exploratory Data Analysis |
| **2** | Produce **feature-importance estimates** from a trained predictive model. The target column is `outcome` — use **only the numerical columns**. Describe how you would explain the technique(s) to the head of loan sales. | Machine Learning |
| **3** | The table below demonstrates the **coefficients produced from a GLM ElasticNet** on the dataset to predict `outcome`. **Interpret** the table and put together **three recommendations** for the client in the form of one or two PowerPoint slides. | Business Strategy |

> **Time limit:** 3 hours for all three tasks combined.

### GLM ElasticNet Coefficients Provided by the Client

The third task provides a ready-made GLM ElasticNet (binomial) coefficient table. Interpreting it is a core part of the exercise.

| Variable                   | Coefficient |
|:---------------------------|------------:|
| outcome_previous.success   |      0.2101 |
| month.mar                  |      0.0830 |
| days_since_previous        |      0.0453 |
| contact.mobile             |      0.0366 |
| job.retired                |      0.0336 |
| consumer_confidence        |      0.0331 |
| job.full_time_education    |      0.0203 |
| default.no                 |      0.0194 |
| month.jul                  |      0.0106 |
| low_temp                   |      0.0091 |
| default.unknown            |     -0.0181 |
| job.industrial             |     -0.0195 |
| num_contacts               |     -0.0263 |
| month.nov                  |     -0.0279 |
| contact.landline           |     -0.0375 |
| day_of_week.mon            |     -0.0444 |
| forward_rate               |     -0.0477 |
| outcome_previous.failure   |     -0.0652 |
| employment_variation       |     -0.1579 |
| month.may                  |     -0.2845 |
| num_employed               |     -0.5581 |
| (Intercept)                |     -2.4090 |

> *Notes: all categorical variables were one-hot encoded, all variables were centred and scaled, zero-coefficient variables excluded; the GLM uses the **binomial** distribution.*

---

## Dataset

### Overview 

| Attribute | Value |
|-----------|-------|
| **File** | `data/LittleBank_Case_Study.csv` |
| **Shape** | (35000, 24) |
| **Rows** | 35,000 |
| **Columns** | 24 (13 numerical, 10 categorical, 1 target) |
| **Target** | `outcome` — TRUE / FALSE |
| **Positive class rate** | **11.29 %** (3,952 subscribers vs. 31,048 non-subscribers) |
| **Imbalance** | Severe — requires resampling or cost-sensitive training |
| **Potential leakage** | `outcome_previous`, `num_contacts_previous` |

### Feature Descriptions

| Column | Description |
|--------|-------------|
| `month` | Month of latest contact |
| `day_of_week` | Day of latest contact |
| `contact` | Type of communication used (mobile / landline) |
| `num_contacts` | Number of contacts during this telemarketing campaign |
| `days_since_previous` | Days since contact in previous campaign (`-1` if not contacted before) |
| `num_contacts_previous` | Number of contacts in previous campaigns |
| `outcome_previous` | Outcome of previous campaigns |
| `age` | Age in years |
| `marital` | Marital status |
| `job` | Type of job |
| `education` | Education level |
| `default` | Is currently in default on a credit product |
| `mortgage` | Has a mortgage |
| `personal_loan` | Has personal loans |
| `call_centre_volume` | Index of load on call centre at time of contact |
| `high_temp` | Recorded high temperature at customer location on the contact day (°C) |
| `low_temp` | Recorded low temperature at customer location on the contact day (°C) |
| `forward_rate` | 3-month forward rate |
| `num_employed` | Number of employees (macroeconomic indicator) |
| `consumer_confidence` | Consumer confidence index at time of contact |
| `price_index` | Weighted average prices of goods |
| `employment_variation` | Relative employment variation over time |
| `outcome` | **Target** — whether the customer subscribed |

---

## Repository Structure

### Directory Tree

```
Data-Science-Case-Study/
│
├── LittleBank_Case_Study.ipynb              # Exhaustive EDA solution (Task 1)
├── LittleBank_Case_Study_ML.ipynb           # Exhaustive ML solution (Task 2)
├── littlebank_case_study.py                 # Script equivalent of the EDA notebook
├── littlebank_case_study_ml.py              # Script equivalent of the ML notebook
│
├── interview_solutions/
│   ├── task_1_eda/
│   │   ├── LittleBank_Case_Study_Concise_Solution.ipynb   # Concise EDA (interview use)
│   │   └── littlebank_case_study_concise_solution.py
│   └── task_2_machine_learning/
│       ├── LittleBank_Case_Study_ML_Concise_Solution.ipynb # Concise ML (interview use)
│       └── littlebank_case_study_ml_concise_solution.py
│
├── data/
│   └── LittleBank_Case_Study.csv            # Source dataset (35,000 records)
│
├── models/
│   └── model_decision_tree.pkl              # Saved Decision Tree model
│
├── images/
│   ├── task_1_figures/                       # EDA visualisations
│   └── task_2_figures/                       # ML visualisations
│
├── save/                                     # Preprocessed data (parquet, gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

### Folder Reference Table

| Folder / File | Purpose |
|---------------|---------|
| `LittleBank_Case_Study.ipynb` | The **detailed** EDA notebook — every section ends with a written business insight. |
| `LittleBank_Case_Study_ML.ipynb` | The **detailed** ML notebook — baselines, regularised models, tree models, XGBoost and feature importance. |
| `interview_solutions/` | **Concise** versions of both notebooks, optimised for the 3-hour time limit. |
| `data/` | The source CSV provided by the client. |
| `models/` | Serialised model artefacts. |
| `images/` | All plots generated during the analysis, split by task. |
| `save/` | Preprocessed train/test parquet files (not tracked in git). |

---

## Brief and Detailed Solutions

This repository provides **two solution tiers** for each task. They exist for very different purposes.

| | Detailed Solution | Concise (Brief) Solution |
|-|-------------------|--------------------------|
| **Purpose** | In-depth understanding and portfolio showcase | Interview time-pressure practice |
| **Location** | `LittleBank_Case_Study.ipynb` / `LittleBank_Case_Study_ML.ipynb` | `interview_solutions/task_1_eda/` / `interview_solutions/task_2_machine_learning/` |
| **Depth** | Exhaustive — written insight after every section | Focused on the essential steps only |
| **Length** | ~140 cells per notebook | Designed to fit comfortably within a 3-hour window |
| **Best for** | Building a complete understanding of the techniques | Timed mock interviews |

> **Recommended workflow.** Study the detailed solution first to fully absorb the data and techniques. Then practise with the concise solution under realistic time pressure until you can consistently finish within the 3-hour limit.

---

## Structure of the Notebooks

### Task 1 — EDA Notebook (`LittleBank_Case_Study.ipynb`)

| Section | Content |
|---------|---------|
| **0.** Load and Overview | Load CSV, inspect dtypes, value counts, class distribution |
| **1.** Data Cleaning | Remove duplicates, audit missing data, handle `"unknown"` categories |
| **2.** Macroeconomic & Environmental | Forward rate, consumer confidence, employment, price index |
| **3.** Day & Month Influence | Success rate by month and day-of-week |
| **4.** Marketing Campaign Analysis | Contact type, number of contacts, campaign's outcome |
| **5.** Customer Profile | Age, job, education, marital status, mortgage, loans |
| **6.** Job Category Deep Dive | Subscription rates broken down by profession |

### Task 2 — ML Notebook (`LittleBank_Case_Study_ML.ipynb`)

| Section | Content |
|---------|---------|
| **0.** Load and Overview | Same as Task 1 |
| **1.** Data Cleaning | Encode months/days, drop `"unknown"` and insignificant columns, remove outliers |
| **2.** Distribution & Correlation | Histograms and correlation heatmap |
| **3.** Class Imbalance & Train/Test Split | Tackle class Imbalance and split the data |
| **3.1** Class Imbalance (SMOTE) | Apply Synthetic Minority Oversampling Technique |
| **3.2.** Train–Test Split | 80 / 20 split |
| **4.** Pre-processing | MinMax scaling + One-Hot Encoding |
| **5.** Save | Export preprocessed data to parquet |
| **6.** Baseline Models | Implement baseline models |
| **6.1** Naive Baseline | Random-guess and all-negative baselines |
| **6.2.** Logistic Regression | GridSearchCV hyper-parameter tuning |
| **7.** Lasso & ElasticNet | L1 and L1+L2 regularisation |
| **8.** Decision Tree & Random Forest | Tree-based models with hyper-parameter tuning |
| **9.** XGBoost | Gradient boosting |
| **10.** Model Saving | Export best model with `joblib` / `pickle` |

---

## Solutions

### Task 1 — Exploratory Data Analysis (EDA)

#### 0. Load and Overview

The first step is to examine understand what are the main characteristics of the data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/LittleBank_Case_Study.csv").convert_dtypes()
df["outcome"] = df["outcome"].astype("string")

print(df.shape)     # (35000, 24)
df.info()
df.describe()
```

**Outcome Class distribution — the single most important finding:**

```python
df["outcome"].value_counts().plot(kind="pie", startangle=90, autopct="%1.2f")
plt.title("Outcome of the Customer's Subscription")
plt.ylabel("")
plt.show()
```

<p align="center">
  <img src="images/task_1_figures/outcome_of_the_customers_subscription.png" width="460" alt="Outcome pie chart"/>
</p>

> **Insight.** Only **11.29 %** of the 35,000 contacted customers subscribed. This severe class imbalance is the most important data characteristic — it dictates the evaluation metric (recall, not accuracy) and the modelling strategy (resampling) used in Task 2. It also signals that the campaign has been largely ineffective: ~31,000 contacts were made with no sale.

---

#### 1. Data Cleaning

```python
# Duplicate check
print(f"Duplicate rows: {df.duplicated().sum()}")

# Missing values are encoded as the literal string "unknown"
for col in df.select_dtypes(include="string").columns:
    n_unknown = (df[col] == "unknown").sum()
    if n_unknown > 0:
        print(f"{col}: {n_unknown} unknowns ({n_unknown/len(df)*100:.1f}%)")

# 'days_since_previous == -1' encodes 'never contacted before'
print(f"No previous contact: {(df['days_since_previous'] == -1).sum()}")
```

**Key cleaning decisions:**

- No duplicate rows found.
- Several categorical columns contain `"unknown"` — treated as a separate category during EDA, dropped in the ML pipeline.

---

#### 2. Macroeconomic and Environmental Factors

The dataset includes macroeconomic indicators (`num_employed`, `employment_variation`, `consumer_confidence`, `forward_rate`, `price_index`) and weather indicators (`high_temp`, `low_temp`) captured at the time of each contact. The analysis ot these factor doesn't provide us with informative insights about the campain. They aren't very helpful for Task 1. However, the ML models use them as one of the most important features that could predict the success of the call. 

---

#### 3. Day and Month Influence
This is the most important analysis. It should be done first. The analysis shows the failure of the marketing campaign.

```python
# Success rate by month
df_outcome_pivot = df_tr.pivot_table(index = "month",
                                     columns = "outcome",
                                     aggfunc = "size",
                                     sort = False)
df_outcome_pivot["Total"] = df_outcome_pivot.iloc[:, 0] + df_outcome_pivot.iloc[:, 1]
df_outcome_pivot["Success_Rate"] = (df_outcome_pivot.iloc[:, 0] / df_outcome_pivot.iloc[:, 2]) * 100

df_outcome_pivot
```

```python
df_outcome_pivot.iloc[:, 0:3].plot(kind = "bar", figsize = (10, 8), rot = 45)

plt.xlabel("Month")
plt.ylabel("Number of Subscriptions")
plt.title("Success of the Marketing Campaign by Months")
plt.show()
```


<p align="center">
  <img src="images/task_1_figures/success_of_the_marketing_campaign_by_months.png" width="680" alt="Success rate by month"/>
</p>

> **Insight.** The **number of subscriptions** for each month is **super low**. Most of the people do not subscribe for the saving account, despite the large number of calls made in May, June, July and August.

---

#### 4. Marketing Campaign Analysis

<p align="center">
  <img src="images/task_1_figures/outcome_of_the_marketing.png" width="640" alt="Outcome of the marketing campaign"/>
</p>

<p align="center">
  <img src="images/task_1_figures/the_outcome_of_the_advertisement_campaign.png" width="640" alt="Advertisement campaign outcome"/>
</p>

**A Masterclass in Failure: When Marketing Goes Horribly Wrong**

- The marketing campaign was a total failure. Both figures show that the subscriptions do not increase, even though thousands of contacts were made. For May, the savings account product was advertised to 11721 people. Only 749 of them have subscribed. This is exactly 6.39%. That's a disaster.

- When we compare the results of the campaign by days of each month, the conclusion is the same - the marketing crashed and burnt. There are days when the success rate is above 50%, but a small number of people subscribed in absolute terms. The hypothesis that there is a day in the month when the subscription rate is relatively high could not be accepted.

---

#### 5. Customer Profile

<p align="center">
  <img src="images/task_1_figures/number_of_customers_by_age.png" width="640" alt="Customer age distribution"/>
</p>

<p align="center">
  <img src="images/task_1_figures/subscriptions_by_proffesion_of_the_customers.png" width="700" alt="Subscriptions by profession"/>
</p>

> **Insight.** **The most customers that subscribed** for the savings account after the marketing campaign have age between 25 and 40. However, this is due to the fact that most of the calls were made to people in that age span. The plot of the success rate based on the customers job shows that those who are in full time education or retired are very likely to subscribe for the savings account product. The success rate of the calls respectively is approx. 45% and 35%. Above 15% success rate of the advertisment can be observed for administrative jobs and unemployed. 

---

### Task 2 — Machine Learning

#### 1. Data Loading and Cleaning

For the ML pipeline, categorical months and days are encoded as integers and `"unknown"` rows are dropped. In addition, the outlier are removed and the outcome target column is transformed into numerical column.

```python
days_of_the_week = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
months_in_the_year = ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug","sep", "oct", "nov", "dec"]

# A dictionary that maps the day with the corresponding number of the day in a week
days_of_the_week_dict = {day:i for i, day in enumerate(days_of_the_week, start = 1)}

# A dictionary that maps the month with the corresponding number of the month in the year
months_in_the_year_dict = {month_str:i for i, month_str in enumerate(months_in_the_year, start = 1)}

# Remove unknown rows for marital feature
df_tr.drop(index = (df_tr[df_tr["marital"] == "unknown"]).index, inplace = True)
```

```python
# Encoding the target column - Outcome

from sklearn.preprocessing import LabelEncoder

target = df_tr["outcome"]

enc_labels = LabelEncoder()
df_tr["outcome_encoded"] = enc_labels.fit_transform(target)
```

---

#### 2. Distribution and Correlation

Histograms of the numerical features and a correlation heatmap are drawn. The goal is to detect skew, multicollinearity and obvious outliers. It worth mentioning that several macroeconomic variables (`num_employed`, `employment_variation`, `forward_rate`) are highly correlated.

The figure bellow depict the distribution of the target value over the months and the days.

<p align="center">
  <img src="images/task_2_figures/month_vs_outcome.png" width="640" alt="Month vs outcome"/>
</p>

<p align="center">
  <img src="images/task_2_figures/days_of_week_vs_outcome.png" width="640" alt="Day of week vs outcome"/>
</p>

> **Insight.** The month-vs-outcome and day-vs-outcome figures outline the data imballance problem that should be taken into account. 

---

#### 3. Handling Class Imbalance and Creating Train/Test Split

The target column `outcome` consist only 11% of people that subscribed for the savings. This means that the data for the LittleBank campaign is imbalanced. The smaller class with customers that didn't subscribe will be often misclassified. The models will have poor performance and won't classify the potential subscribers. The SMOTENC (Synthetic Minority Oversampling Technique for Nominal and Continuous data) is applied to address the problem.

```python
from imblearn.over_sampling import SMOTENC

smote = SMOTENC(categorical_features = categorical_cols_df, sampling_strategy = "auto", random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_resampled.shape, y_resampled.shape
```

```
((56050, 20), (56050,))
```

> **Why SMOTENC?** It improves the classification of the minority class. Another benefit of SMOTENC is the improvement of the ML model's variance. A disadvantage of the approach is the computational cost of the technique. However, in our case, it could be omitted because the data is relatively small.

Next, the data is splitted into 80% train data and 20% test data.

```python
train_df, test_df = train_test_split(df_tr_resampled, test_size = 0.2, random_state = 42)

train_df.shape, test_df.shape
```
```
((44840, 21), (11210, 21))
```

---

#### 4. Pre-processing

Two main pre-processing techniques are applied - scaling (`MinMaxScaller`) and Encoding (`OneHotEncoding`)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(df_tr[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
```

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output = False, handle_unknown = "ignore")
encoder.fit(df_tr[categorical_cols])
```

> **Task 2 constraint.** The client specifically requires **only the numerical columns** to be used for the feature importance model. On the other hand, the solution applies most of the features. This is due to the fact that one of the goals of the project is to solve the task comprehensively. However, in an interview situation, the instructions of the interviewers should be followel.

---

#### 5. Model Training and Evaluation

Six models are train to solve Task 2 in ascending complexity. Each model has its own strengths and limitations. They are listed below:

1. **Baselines** (random-guess and all-negative) — create expectations and set thresholds.
2. **Logistic Regression** with `GridSearchCV`.
3. **Lasso** (L1) and **ElasticNet** (L1 + L2) regularised logistic models.
4. **Decision Tree** — with and without pruning.
5. **Random Forest** — the best results, with top accuracy and highest precision.
6. **XGBoost** — close second, but highly complex and task unefficient.

The Random Forest model implementation with fine-tuned parameters is given below:


```python
from sklearn.ensemble import RandomForestClassifier

model_random_forest = RandomForestClassifier(n_jobs = -1,
                                             n_estimators = 500,       
                                             max_features = 9,          
                                             max_depth = 35,            
                                             min_samples_split = 2,
                                             random_state = 42)

model_random_forest.fit(train_inputs, train_targets)

train_preds_rforest = model_random_forest.predict(train_inputs)
test_preds_rforest = model_random_forest.predict(test_inputs)

train_acc = accuracy_score(train_targets, train_preds_rforest)

test_acc, recall_test, precision_test = calculate_basic_metrics(test_targets, test_preds_rforest)

print(f"Train Accuracy = {train_acc:.5f}, Test Accuracy = {test_acc:.5f}")
print(f"Recall = {recall_test:.5f}, Precision = {precision_test:.5f}")
```

```
Train Accuracy = 1.00000, Test Accuracy = 0.89590
Recall = 0.90316, Precision = 0.88792
```
---

#### 6. Feature Importance

The essence of the second task is to **produce estimates of feature importance from a trained predictive model**. In other words, we have to train the model and find out how it makes decision to classify customers aa potential subscriber. Not just that, but we also have to figure out the factors that influence person's decision and give insights to the head of loan's sale, based on the results. After all, the bank invested thousands of dollars to advertise and increase the subscribtions to the classic savings account.

```python
importance_df = pd.DataFrame({
    "feature": train_inputs.columns,
    "importance": model_random_forest.feature_importances_
}).sort_values("importance", ascending = False)

sns.barplot(data = importance_df.head(15), x = "importance", y = "feature",
            hue = "feature", legend = False, palette = "tab10")

plt.title("Feature Importance of the Random Forest Model");
```

Plots of the feature importance were generated for all of the models. However, the empasis will be put on the best performing model.

<p align="center">
  <img src="images/task_2_figures/feature_importance_random_forest.png" width="680" alt="Random Forest feature importance"/>
</p>

> **Insight.** The decision tree distinguish the `forward_rate`, `num_employed` and `employment_variation` as one of the most important features. The macroeconomic features dominate top 10 ranking. Alternatively, the model considers environmental factors like `low_temperature` and `high_temperature` connected to the decision of a customer to subscribe for the product. The `age` of the clients also influence the success of the telemarketing. It is significant to point out - the `call_centre_volume` also impact significantly the decision of the model. However, based on the analysis for Task 1, the load of the call centre is very low to have any a big impact. It is crucial to highlight that the factors for the unsuccessful campaign should be found in a different place.

**How to explain Random Forest to a non-technical stakeholder:**
> *"We train hundreds of small decision trees using a random slice of customers and features. Every tree makes its own prediction as to whether a customer will subscribe. The final decision comes from a majority vote across all trees. Feature importance reflects how frequently and how effectively trees distinguish between subscribers and non-subscribers."*

---

### Task 3 — Business Strategy / Recommendations

Based on the [table](#glm-elasticnet-coefficients-provided-by-the-client) that Little bank provided, the data shows:

1. If the previously campaign was successful, there is a high probability that the client would subscribe again for the product.

2. If the contact was made in March the chances for subscription are high. For July, there is a little chance of a positive outcome, in November – low chance of a positive outcome. Nevertheless, in May we could say that most certainly the customer won’t subscribe to the product.

3. People who are retired or in full-time education are more likely to purchase the product than people working in the industry. 

To improve the conversion rates it is recommended to:

1. Make a campaign focused on the right customer segment, with an accurate message, via the most appropriate medium (Social media, Website, not via Mobile or Landline).

2. Find the true needs of the customers and personalise the product for them. An improvement of the product will be needed in order to increase the conversion rates. 

3. To get feedback from the bank clients, do they like the products of the bank and what makes them eager to purchase a product. Satisfied client is more likely to purchase a product or recommend it to a friend.

> For the presentation's bullets or more insights/recommendation, check the **add a link to the folder** (the link..)[linking_park]

---

## Models and Accuracy

Six model families were trained and evaluated on the data. The results are taken directly from the ML notebook and are shown below.

| Model                     | Train Accuracy | Test Accuracy | Recall    |  Precision  |
| ------------------------- |:--------------:|:-------------:|:---------:|:-----------:|
| Logistic Regression       | 0,7512         | 0,7604        | 0,7239    | 0,7758      |
| Lasso Regression          | 0,7519         | 0,7603        | 0,7223    | 0,7766      |
| ElasticNet                | 0,7515         | 0,7602        | 0,7232    | 0,7759      |
| Decision Tree             | 0,8942         | 0,8346        | 0,8482    | 0,8224      |
| **Random Forest** 🏆🏆🏆 | **1.00000**    | **0,8959**  | **0,9032**  | **0,8879**  |
| Extreme Gradient Boosting | 0,9998         | 0,8954        | 0,9097    | 0,8821      |

> **Top performing model.** Withouth a doubt, the best results are achieved with the Random Forest model. It has the highest `test accuracy` and the second highest `recall`.

> **Primary metric: `Recall`.** In the business context, missing a genuine subscriber (false negative) is costlier than contacting a non-subscriber (false positive). The `Recall` metric measures this. It is the proportion of true subscribers that the model successfully identifies.

---

## Conclusion and Key Insights

### 📢🎯 Marketing Campaign Performance 

- 🚨 **Failure of the Marketing Campaign.** The failure of the Marketing Campaign is very clear. There isn't any impactful improvement of the subscriptions for classic savings account. Despite of the advertising. As an example 11,721 contacts were made in May. Only 749 customers subscribed. 6.39% conversion rate. → extremely low conversion

- ↑↓ **High activity ≠ High impact.** Large number of calls did not translate into increase of subscriptions, indicating poor targeting or messaging. Some months show ~50% success rates, but on very small number of subscribers. → high conversion months lack scale

- 🚀 **Business Implications.** Current strategy is not scalable and increasing calls won't lead to higher conversion rate.

### ☎️📲 Contact Strategy & Call Behavior

- 📲 **Mobile Dominates oOutreach.** The majority of contacts were made via mobile - 22,215 mobile calls vs 12,785 landline calls. → both are outdated and ineffective approaches.

- ❌ **Severe Contact Inefficiency.** There are extreme outliers in the data (e.g., 40+ calls to a single customer). That suggests poor contact strategy. → spam-like patterns harming the brand

- ✅ **Successful Contact Strategy** The average number of contacts for a successful conversion is ~2 calls. The successful second calls occur ~6 days after the first contact. → the optimal contact strategy

- 🚀 **Business Implications.** Optimise call strategy increases the conversion rates. However, the marketing strategy is heavily outdated. The strategy should expand into digital channels.

### 👦👧 Customer's Profile 

- 💪 **High-performing Customer Segments.** - Most conversions are achieved amongst the Administrative professionals, Retired individuals, Unemployed customers and Students (full-time education). → more focused campaing towards the right segment

- 🎓 **Education Level.** Majority of customers (both overall and subscribers) have secondary or higher education. → difference in education do not influence results

- 📅 **Age.** The campaign focuses on 20–60 age range. The retired individuals are part of the high-performing segments, but are neglected. → missed oportinities with retired individuals

- 🏡 **Mortgage.** Mortgage holders are more likely to convert. Over 50% of subscribers (2133 / 3952) have a mortgage → mortage is strong indicator for conversions

- 💍 **Personal Loans and Marital Status.** Personal loans negatively correlate with conversion. Only 573 / 3952 subscribers have personal loans. Marital status has no impact over the subscription → weal indicators for conversions

- 🚀 **Business Implications.** More focus on the higher performing customer segments. A targeted survey is mandatory. It could uncover why one segments are converting high and other don't.

### 📊📈 Model Performance Summary 

- 📈 **Linear Models Plateau.** Logistic Regression, Lasso, and ElasticNet reach ~76% test accuracy. → Non-linear patterns in data.

- 🖧 **Decision Trees - Better Option.** Decision Tree improves the accuracy but overfits. The model achieves 83.46% test accuracy, but with a clear train–test gap (89.42% vs 83.46%). → limited generalization

- 🌲🌳🌿 **Tree-based Ensembles Dominance.** Random Forest and XGBoost outperform by far all other models . Random Forest hits the best precision (88.79%) and accuracy (89.59%). XGBoost has the highest recall 90.97% and near-best accuracy 89.54%. → ensemble models are the best solution for the problem

- 🚀 **Business Implications.** Random Forest and XGBoost are the best choice when the goal is to reduce the wasted outreach and maximise the subscriber detection. Note, overfitting and computation eficiency should be taken into consideration. Less complex models could be used for better understanding of the data and classification decision.

---

## How to Tackle the Interview

This is the section you came here for. It is split into three parts:

1. **Basic advice** for a Data Science / ML interview at a Big Four firm.
2. **Task-1 specific advice** — how to approach the EDA task.
3. **Task-2 specific advice** — how to approach the ML task.

> This section will continue to be expanded with additional tips, failure modes to avoid and LLM-assisted workflows. The core framework is below.

---

### Basic Advice for a Big Four Data Science / ML Interview

**Know the environment.**

- Big Four interview panels frequently include **non-technical stakeholders** (engagement managers, partners). Every explanation must land for a smart business audience, not just a machine-learning peer.
- There are usually **3–5 interviews** in total: HR screen, technical case study (this one), business-case presentation, a partner / fit interview and sometimes a modelling deep-dive.
- The **technical case study** is typically **3 hours** for all questions combined and is done on your own machine.

**Prepare your environment.**

- Have a clean Jupyter/VS Code setup ready with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn` and `xgboost` pre-installed.
- Keep a **personal snippets file** with your favourite EDA boilerplate (`df.info()`, `df.describe()`, missing-value audit, class-imbalance check).
- Have the dataset loaded and the concise notebook template open before the clock starts.

**Use a repeatable answer structure.**

- For every question, use **Problem → Approach → Result → Recommendation**. Big Four reviewers score you on structured thinking as much as on code quality.

**Time-box aggressively.**

| Phase | Suggested budget |
|-------|:-----------------|
| Data overview + cleaning | 20 min |
| Core EDA with 5–7 plots | 60 min |
| ML pipeline (baseline → RF / XGB) | 60 min |
| Feature importance + interpretation | 20 min |
| Business recommendations | 20 min |

> My original case-study attempt failed **specifically because I ran out of time**. The single most impactful change was switching to the concise notebook template and practising until I could finish within 3 hours reliably.

**Use LLMs strategically.**

- Use Claude, ChatGPT, or Copilot to generate boilerplate quickly — but **understand every line** you submit.
- Good prompts: *"Write a function that computes subscription rate by month and plots it as a bar chart."*
- Bad prompts: *"Solve this case study."* — you will fail when asked to explain your code.
- Keep a prepared library of prompts for: data-overview, class-imbalance handling, SMOTE, GridSearchCV, feature-importance plotting.

**Rehearse the narrative.**

- Record yourself walking through the notebook as if presenting to the head of loan sales. Focus on *why* each step exists, not *what* you typed.

---

### Advice for Task 1 — Exploratory Data Analysis

**Start with the three-command overview.**

```python
df.info()
df.describe(include="all")
df["outcome"].value_counts(normalize=True)
```

These three lines reveal dtypes, scale, missing data and the class distribution — roughly **80 % of everything you need to know** about the dataset.

**Flag class imbalance immediately.**

- Say it out loud: *"The positive class is only 11.3 %. This has three consequences: (i) I'll use recall as the headline metric, (ii) I'll apply SMOTE inside the training fold and (iii) I'll include a naive all-negative baseline to anchor expectations."*

**Choose 5–7 impactful visualisations — not 15 mediocre ones.**

For this dataset, the must-have plots are:

1. Target class distribution (pie or bar).
2. Success rate by month.
3. Success rate by day-of-week.
4. Success rate by profession.
5. Correlation heatmap of numerical features.
6. Histogram of `num_contacts` split by outcome.
7. (Bonus) Economic indicator trend vs. outcome.

**Always close each section with a one-sentence business insight.**

Not: *"May has the lowest count."*
Yes: *"May is the worst month for conversion — we should cut campaign spend in May by 70 %."*

**Handle missing data transparently.**

- `"unknown"` is **not** a missing value in a strict sense — it is a category. Treat it as such in EDA. In the ML pipeline, drop it or one-hot encode it explicitly.

---

### Advice for Task 2 — Machine Learning

**Always start with a baseline.**

```python
# All-negative baseline: predict FALSE for everyone.
y_pred_baseline = np.zeros_like(y_test)
print("Baseline accuracy:", (y_pred_baseline == y_test).mean())   # ~88.7 %
print("Baseline recall  :", 0.0)                                  # zero subscribers found
```

This single block disarms the "why not just predict majority class?" trap and shows rigour.

**Explain class imbalance and your solution before training.**

State clearly: *"Because the positive rate is only 11 %, I'll apply SMOTE **only inside the training fold** to avoid data leakage and I'll evaluate using recall and precision — not raw accuracy."*

**Match the metric to the business.**

- **Missing a subscriber (false negative) costs more than contacting a non-subscriber (false positive)** — recall is therefore the primary metric.
- Be prepared to defend this with a simple cost sketch: *"A missed subscriber is a lost lifetime-value of ~£X; an unwanted call costs a few pence of call-centre time."*

**Climb the model-complexity ladder.**

1. Logistic Regression — fast, interpretable, benchmark.
2. Lasso / ElasticNet — regularisation + automatic feature selection.
3. Decision Tree — the interpretability sweet-spot.
4. Random Forest — the workhorse.
5. XGBoost — the state-of-the-art finisher.

**Prepare plain-English one-liners for every model.**

- *Logistic Regression*: "A weighted vote over the features — weights tell us direction and strength of influence."
- *Random Forest*: "A committee of decision trees, each trained on a random subset of the data — we take the majority vote."
- *XGBoost*: "Many shallow trees built sequentially, each correcting the mistakes of the previous one."

**Feature importance is a storytelling tool.**

- Sort the bars from largest to smallest and read them as a business narrative: *"The model tells us timing and economic context matter most; the customer's profession and recent contact history matter second; demographics matter least."*

**Use the task constraint ("only numerical columns") as a strength.**

- Do not argue with the constraint — comply, but **mention** that in production you would retrain on all features and expect a meaningful lift. This shows judgement without defying the brief.

---

## Python Scripts

The `.py` files are auto-generated script equivalents of the Jupyter notebooks, suitable for batch execution, CI pipelines, or quick command-line re-runs.

| Script | Notebook Equivalent | Description |
|--------|---------------------|-------------|
| `littlebank_case_study.py` | `LittleBank_Case_Study.ipynb` | Full EDA pipeline — sections 0 through 6 |
| `littlebank_case_study_ml.py` | `LittleBank_Case_Study_ML.ipynb` | Full ML pipeline — cleaning through model saving |
| `interview_solutions/task_1_eda/littlebank_case_study_concise_solution.py` | Concise EDA notebook | Streamlined EDA for time-constrained practice |
| `interview_solutions/task_2_machine_learning/littlebank_case_study_ml_concise_solution.py` | Concise ML notebook | Streamlined ML pipeline for time-constrained practice |

To run any script:

```bash
python littlebank_case_study.py
```

---

## Requirements

Install all dependencies in a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Key libraries used in the analysis:**

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical computing |
| `matplotlib` | ≥ 3.7 | Plotting |
| `seaborn` | ≥ 0.12 | Statistical visualisation |
| `scikit-learn` | ≥ 1.3 | ML models, preprocessing, metrics |
| `imbalanced-learn` | ≥ 0.11 | SMOTE for class imbalance |
| `xgboost` | ≥ 1.7 | Gradient boosting |
| `pyarrow` | ≥ 13.0 | Parquet file support |
| `joblib` | ≥ 1.3 | Model serialisation |
| `jupyter` | ≥ 1.0 | Notebook runtime |

---

## Contributing

Contributions, suggestions and improvements are very welcome — the repository is updated regularly.

**To contribute:**

1. **Fork** the repository.
2. **Create** a feature branch (`git checkout -b feature/your-feature-name`).
3. **Commit** your changes (`git commit -m "Add: your feature description"`).
4. **Push** to the branch (`git push origin feature/your-feature-name`).
5. **Open** a Pull Request — PRs are welcome!

For major changes, please open an issue first to discuss what you would like to change.

**Good first contributions:**

- Additional interview tips and "gotcha" questions from your own Big Four experience.
- Alternative modelling approaches (CatBoost, LightGBM, neural models).
- More visualisations for the EDA notebook.
- Translations of the README into other languages.

**Maintainer:** [@krasvachev](https://github.com/krasvachev) — feel free to open an issue or reach out directly.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for the full text.

You are free to use, copy, modify, merge, publish, distribute and adapt this work, including for commercial purposes, as long as the original author is credited.

---

<p align="center">
  <em>Good luck with your interview. You've got this. 🎯🎯🎯</em>
  <br><br>
  <strong>If this repository helped you, please consider giving it a ⭐ — it helps other candidates find it.</strong>
</p>
