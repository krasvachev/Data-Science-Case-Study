# LittleBank — Data Science Case Study

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931a?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-006400)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
[![Last Commit](https://img.shields.io/github/last-commit/krasvachev/Data-Science-Case-Study)](https://github.com/krasvachev/Data-Science-Case-Study)
[![Stars](https://img.shields.io/github/stars/krasvachev/Data-Science-Case-Study?style=social)](https://github.com/krasvachev/Data-Science-Case-Study)

> **A complete, interview-ready Data Science case study for Big Four (Deloitte, EY, KPMG, PwC) technical interviews — featuring an exhaustive EDA, a full ML pipeline, business-focused recommendations, and a detailed interview preparation guide.**

---

## Introduction

A Data Scientist or Machine Learning Engineer role at one of the Big Four (Deloitte, EY, KPMG, PwC) is a great opportunity that should not be missed. The Big Four companies generate $219 billion in revenue, with worldwide offices and more than 1.5 million employees. This makes them the largest professional service and accounting companies in the world. Being a part of them means you could work on a variety of projects across a wide range of fields, and build broad experience across many topics — especially in the early stages of a career. However, to be a part of the Big Four, first you have to pass the interview process. And the case study task... I mean the technical case study.

What is a case study? That is the question I asked myself when I first heard that I had to tackle such a problem. Typically, there are between 3 and 5 interviews when applying to a Big Four firm. The case study is a business task that the candidate has to solve within a fixed time. For the tech interview, the company often provides a real-world dataset. The goal is to perform Exploratory Data Analysis (EDA), apply Machine Learning (ML) models, and answer specific business questions connected to the task. Usually, the time to solve all three tasks is 3 hours.

When I was doing my preparation, I could not find a case study example for Data Scientists or ML Engineers. This is the first reason for creating this repository — to provide job seekers with an exercise to practise on. There is also a second reason: I failed to pass the data science case study. However, I decided to solve the task outstandingly with a solution that successfully passes the case study interview. And last but not least, the goal of the repo is not just helping me, but also helping **you** land that dream job.

The repo provides an exhaustive solution to a Data Science case study task given during a Big Four technical interview. It also provides guidance on how to prepare for the interviews. The main focus is put on how to tackle the case study. There are also hints on how to use LLM models to help you efficiently during your interview preparation.

The solutions are not limited to the Big Four accounting companies. They can be helpful for other accounting and professional service firms as well. In addition, the solutions apply the most common ML and Data Science practices. That is why they can be a great resource for any ML Engineer or Data Scientist preparing for a tech interview.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [The Task](#the-task)
3. [Dataset](#dataset)
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

## Repository Structure

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

| Folder / File | Purpose |
|---------------|---------|
| `LittleBank_Case_Study.ipynb` | The **detailed** EDA notebook — every section ends with a written business insight. |
| `LittleBank_Case_Study_ML.ipynb` | The **detailed** ML notebook — baselines, regularised models, tree models, XGBoost, and feature importance. |
| `interview_solutions/` | **Concise** versions of both notebooks, optimised for the 3-hour time limit. |
| `data/` | The source CSV provided by the client. |
| `models/` | Serialised model artefacts. |
| `images/` | All plots generated during the analysis, split by task. |
| `save/` | Preprocessed train/test parquet files (not tracked in git). |

---

## The Task

### Customer Analytics Case Study — Cross-sell Opportunities for LittleBank

> **Client:** LittleBank — a retail bank providing deposit accounts, loans, and savings products.
>
> **Problem:** The head of loan sales has noticed a recent **drop in subscriptions of the "classic savings account"** product, despite consistent telemarketing efforts offering the account to customers. He has turned to our company for advice on how to improve sales of this product.

LittleBank has shared a data file (`LittleBank_Case_Study.csv`) containing historical telemarketing-campaign records. The file includes:

- (i) Attributes of customer contacts when a classic savings product was offered.
- (ii) Details of any previous campaigns where a similar product had been offered.
- (iii) Customer attributes.
- (iv) Macroeconomic and environmental factors at the time each contact was made.
- (v) An indicator variable showing whether or not the client bought the product.

Since LittleBank has not yet used advanced analytics in its sales and marketing activities, the candidate must come prepared to **describe every algorithm employed and the approach taken** in plain, non-technical language.

### Business Questions

| # | Task | Type |
|---|------|------|
| **1** | What steps would you take to **understand and clean this data**? Perform **Exploratory Data Analysis (EDA)**. | Data Analysis |
| **2** | Produce **feature-importance estimates** from a trained predictive model. The target column is `outcome` — use **only the numerical columns**. Describe how you would explain the technique(s) to the head of loan sales. | Machine Learning |
| **3** | The table below demonstrates the **coefficients produced from a GLM ElasticNet** on the dataset to predict `outcome`. **Interpret** the table and put together **three recommendations** for the client in the form of one or two PowerPoint slides. | Business Strategy |

> **Time limit:** 3 hours for all three tasks combined.

### GLM ElasticNet Coefficients Provided by the Client

The third task provides a ready-made GLM ElasticNet (binomial) coefficient table. Interpreting it is a core part of the exercise.

| Variable | Coefficient | | Variable | Coefficient |
|----------|:-----------:|-|----------|:-----------:|
| `outcome_previous.success` | **+0.2101** | | `default.unknown` | −0.0181 |
| `month.mar` | +0.0830 | | `job.industrial` | −0.0195 |
| `days_since_previous` | +0.0453 | | `num_contacts` | −0.0263 |
| `contact.mobile` | +0.0366 | | `month.nov` | −0.0279 |
| `job.retired` | +0.0336 | | `contact.landline` | −0.0375 |
| `consumer_confidence` | +0.0331 | | `day_of_week.mon` | −0.0444 |
| `job.full_time_education` | +0.0203 | | `forward_rate` | −0.0477 |
| `default.no` | +0.0194 | | `outcome_previous.failure` | −0.0652 |
| `month.jul` | +0.0106 | | `employment_variation` | −0.1579 |
| `low_temp` | +0.0091 | | `month.may` | **−0.2845** |
| | | | `num_employed` | **−0.5581** |
| | | | *(Intercept)* | −2.4090 |

> *Notes: all categorical variables were one-hot encoded, all variables were centred and scaled, zero-coefficient variables excluded, and the GLM uses the **binomial** distribution.*

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **File** | `data/LittleBank_Case_Study.csv` |
| **Rows** | 35,000 |
| **Columns** | 24 (12 numerical, 11 categorical, 1 target) |
| **Target** | `outcome` — TRUE / FALSE |
| **Positive class rate** | **11.29 %** (3,952 subscribers vs. 31,048 non-subscribers) |
| **Imbalance** | Severe — requires resampling or cost-sensitive training |

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

## Brief and Detailed Solutions

This repository provides **two solution tiers** for each task. They exist for very different purposes.

### Why Two Solutions?

| | Detailed Solution | Concise (Brief) Solution |
|-|-------------------|--------------------------|
| **Purpose** | Deep learning and portfolio showcase | Interview time-pressure practice |
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
| **2.** Macroeconomic & Environmental | Forward rate, consumer confidence, employment, price index, temperature |
| **3.** Day & Month Influence | Success rate by month and day-of-week |
| **4.** Marketing Campaign Analysis | Contact type, number of contacts, previous-campaign outcomes |
| **5.** Customer Profile | Age, marital status, job, education |
| **6.** Job Category Deep Dive | Subscription rates broken down by profession |

### Task 2 — ML Notebook (`LittleBank_Case_Study_ML.ipynb`)

| Section | Content |
|---------|---------|
| **0.** Load and Overview | Same as Task 1 |
| **1.** Data Cleaning | Encode months/days, drop `"unknown"`, remove outliers |
| **2.** Distribution & Correlation | Histograms and correlation heatmap |
| **3.** Class Imbalance (SMOTE) | Apply Synthetic Minority Oversampling Technique |
| **3.2.** Train–Test Split | 80 / 20 split |
| **4.** Preprocessing | MinMax scaling + One-Hot Encoding |
| **5.** Save | Export preprocessed data to parquet |
| **6.** Baseline Models | Random-guess and all-negative baselines |
| **6.2.** Logistic Regression | GridSearchCV hyper-parameter tuning |
| **7.** Lasso & ElasticNet | L1 and L1+L2 regularisation |
| **8.** Decision Tree & Random Forest | Tree-based models with hyper-parameter tuning |
| **9.** XGBoost | Gradient boosting |
| **10.** Model Saving | Export best model with `joblib` / `pickle` |

---
