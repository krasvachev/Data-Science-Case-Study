# Task 2 — Machine Learning · Concise Interview Solution

This folder contains the **concise solution** for **Task 2: Machine Learning** of the LittleBank Case Study. It is the version you should actually use during a timed mock interview.

## Files

| File | Purpose |
|------|---------|
| `LittleBank_Case_Study_ML_Concise_Solution.ipynb` | Jupyter notebook — run interactively |
| `littlebank_case_study_ml_concise_solution.py` | Script equivalent — run from the command line |

## Why "Concise"?

The detailed solution in the repository root (`LittleBank_Case_Study_ML.ipynb`) trains six model families and includes exhaustive hyper-parameter tuning. This concise version focuses on the essentials you can realistically produce within the 3-hour window:

- Data cleaning tailored for ML (integer encoding, drop `"unknown"`, IQR outlier removal).
- Class-imbalance handling with **SMOTE** (applied only on the training set).
- Preprocessing with `MinMaxScaler` + `OneHotEncoder` via `ColumnTransformer`.
- A compact model comparison: baseline → Logistic Regression → Random Forest → XGBoost.
- **Feature importance** plot on the numerical columns (as required by the task brief).
- Evaluation on **Recall** (primary), Precision, and Accuracy.

## Key Results (from this pipeline)

| Model | Test Accuracy | Precision | Recall |
|-------|:-------------:|:---------:|:------:|
| Logistic Regression | 0.7604 | 0.7758 | 0.7239 |
| **Random Forest** 🏆 | **0.8959** | **0.9032** | **0.8879** |
| Extreme Gradient Boosting | 0.8954 | 0.9097 | 0.8821 |

> **Primary metric is recall** — we care about finding real subscribers, not maximising overall accuracy.

## How to Use for Interview Practice

1. Open the notebook and **clear all outputs**.
2. Set a **75-minute timer** (the recommended Task 2 budget).
3. Work top-to-bottom without looking at the solution.
4. Be ready to **explain every model in plain English** — Big Four interviewers will probe this.
5. At the end of the timer, compare your output to the reference notebook.
6. Repeat until you can complete it confidently within the time limit.

> **Tip.** Prepare a one-liner for each model before the interview. For Random Forest: *"A committee of decision trees, each trained on a random subset of the data — the forest predicts by majority vote."*

## Related

- [← Back to main README](../../README.md)
- [← Task 1 · EDA concise solution](../task_1_eda/README.md)
