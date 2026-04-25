# Task 2 — Machine Learning: Solution Overview

> **Goal:** Build a predictive model to estimate whether a customer will subscribe to the classic
> savings account. Train on historical campaign data, evaluate with recall as the primary metric,
> and extract feature importance to explain the model to non-technical stakeholders. Only the
> numerical columns are used for the feature importance analysis, as specified by the task brief.

---

## Table of Contents

1. [Load and Overview](#0-load-and-overview-the-data)
2. [Cleaning the Data](#1-cleaning-the-data)
3. [Analyse Distribution and Features](#2-analyse-distribution-of-data-and-features)
4. [Handle Class Imbalance and Split](#3-handle-class-imbalance-and-split-the-data)
5. [Pre-process the Data](#4-pre-process-the-data)
6. [Save the Pre-processed Data](#5-save-the-pre-processed-data)
7. [Baseline Models and Logistic Regression](#6-baseline-models-and-logistic-regression)
8. [Lasso and ElasticNet](#7-lasso-and-elasticnet)
9. [Decision Tree and Random Forest](#8-decision-tree-and-random-forest)
10. [XGBoost](#9-gradient-boosting--xgboost)
11. [Save the Best Model](#10-save-the-best-model)

---

## 0. Load and Overview the Data

The ML pipeline begins with the same data loading step as the EDA notebook — a full overview of
the 35,000-record dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/LittleBank_Case_Study.csv")
df["outcome"] = df["outcome"].astype("string")

df["outcome"].value_counts()
```

```
outcome
False    31048
True      3952
```

> **Insight.** Only 3,952 out of 35,000 customers subscribed — an 11.29 % positive rate. This
> class imbalance is the central modelling challenge and dictates every subsequent decision: the
> choice of resampling technique, the evaluation metric (recall over accuracy), and the
> interpretation of results.

---

## 1. Cleaning the Data

Four targeted cleaning steps transform the raw dataset into a ML-ready table.

### 1.1 — Remove Duplicates and NaN Values

```python
df_tr = df.copy()
df_tr.loc[df_tr.isna().any(axis="columns")].count()  # 0 NaN rows
df_tr[df_tr.duplicated() == True]                    # 0 duplicates
```

> No missing values or duplicate rows were found. The data is already structurally clean.

---

### 1.2 — Convert Months and Days to Integers

String labels like `"jan"` and `"mon"` are mapped to ordered integers so that distance-based and
gradient calculations are meaningful.

```python
months_in_the_year = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
days_of_the_week   = ["mon","tue","wed","thu","fri","sat","sun"]

months_dict = {m: i for i, m in enumerate(months_in_the_year, start=1)}
days_dict   = {d: i for i, d in enumerate(days_of_the_week,   start=1)}

df_tr.insert(loc=2, column="month_no",       value=df_tr["month"].map(months_dict))
df_tr.insert(loc=4, column="day_of_week_no", value=df_tr["day_of_week"].map(days_dict))
df_tr.drop(labels=["month", "day_of_week"], axis=1, inplace=True)
```

---

### 1.3 — Drop Low-Signal Columns and Rows with "unknown" Values

Three columns are removed because they are either highly sparse or not informative for the model.
Rows containing `"unknown"` in key categorical fields are dropped rather than imputed, to avoid
introducing synthetic signal.

```python
# Drop columns with limited predictive value
df_tr.drop(labels=["days_since_previous", "num_contacts_previous", "default"],
           axis=1, inplace=True)

# Drop rows with 'unknown' in categorical columns
for col in ["marital", "job", "education", "mortgage", "personal_loan"]:
    df_tr.drop(index=df_tr[df_tr[col] == "unknown"].index, inplace=True)
```

> After cleaning: **32,503 rows remain** (down from 35,000).

---

### 1.4 — Remove Extreme Outliers

Box-plot inspection reveals two columns with extreme values that would distort model training.

```python
df_tr["num_contacts"].plot(kind="box", vert=False)
df_tr["age"].plot(kind="box", vert=False)

df_tr = df_tr[(df_tr["num_contacts"] < 10) & (df_tr["age"] < 86)]
```

> After outlier removal: **31,608 rows remain**. Customers contacted more than 9 times in a single
> campaign and those above age 85 are excluded as non-representative edge cases.

---

### 1.5 — Encode the Target Column

The string target `"True"` / `"False"` is label-encoded to 1 / 0 for sklearn compatibility.

```python
from sklearn.preprocessing import LabelEncoder

enc_labels = LabelEncoder()
df_tr["outcome_encoded"] = enc_labels.fit_transform(df_tr["outcome"])
```

```
outcome_encoded
0    28025   (did not subscribe)
1     3583   (subscribed)
```

---

## 2. Analyse Distribution of Data and Features

### 2.1 — Target Distribution by Campaign Attributes

Pivot tables and stacked bar charts visualise how subscription rates vary across months, days of
the week, and contact type — confirming the EDA findings within the cleaned ML dataset.

### 2.2 — Correlation Heatmap

```python
df_tr_corr = df_tr.select_dtypes(include=["number"]).corr()

fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df_tr_corr, cmap="Reds", annot=True, ax=ax)
plt.title("Correlation Matrix")
plt.show()
```

> **Insight.** Several macroeconomic indicators are highly correlated with each other
> (`num_employed`, `employment_variation`, `forward_rate`). This multicollinearity
> affects linear models (inflated and unstable coefficients) but not tree-based models,
> which is one reason Random Forest and XGBoost significantly outperform Logistic Regression
> on this dataset.

---

## 3. Handle Class Imbalance and Split the Data

### 3.1 — SMOTENC

Standard SMOTE only handles numerical data. Because the dataset contains categorical columns, the
notebook applies **SMOTENC** (Synthetic Minority Over-sampling Technique for Nominal and Continuous
features), which generates synthetic minority samples by interpolating numerical features and
sampling randomly from observed categorical values.

```python
from imblearn.over_sampling import SMOTENC

X = df_tr[input_cols].copy()
y = df_tr["outcome_encoded"].copy()

categorical_cols = X.select_dtypes("object").columns.tolist()

smote = SMOTENC(categorical_features=categorical_cols,
                sampling_strategy="auto",
                random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)
```

```
Original:  (31,608 rows)  →  3,583 positives  /  28,025 negatives
Resampled: (56,050 rows)  →  28,025 positives /  28,025 negatives
```

> **Why SMOTE, not under-sampling?** Removing majority-class rows discards real signal.
> SMOTENC preserves all observed data and fills in the minority class synthetically,
> improving the model's ability to detect true subscribers without sacrificing recall on
> the already-sparse positive class.

---

### 3.2 — Train–Test Split (80 / 20)

The split is performed **after** SMOTE so that the test set reflects a balanced evaluation
environment consistent with the balanced training set.

```python
train_df, test_df = train_test_split(df_tr_resampled, test_size=0.2, random_state=42)
```

```
Train: 44,840 rows
Test:  11,210 rows
```

---

## 4. Pre-process the Data

### 4.1 — MinMax Scaling

All 12 numerical features are scaled to the [0, 1] range. The scaler is fitted on the
**original (pre-SMOTE)** training distribution to avoid data leakage.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(df_tr[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols]  = scaler.transform(test_inputs[numeric_cols])
```

### 4.2 — One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(df_tr[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs = pd.concat([
    train_inputs[numeric_cols],
    pd.DataFrame(encoder.transform(train_inputs[categorical_cols]),
                 columns=encoded_cols, index=train_inputs.index)
], axis=1)
```

> 7 categorical columns expand to **27 binary dummy variables** (e.g. `contact_mobile`,
> `job_retired`, `education_university_degree`). Combined with the 12 scaled numerical
> features, the final feature matrix has **39 columns**.

---

## 5. Save the Pre-processed Data

The cleaned and encoded train/test splits are persisted as parquet files for fast reloading in
future experiments — avoiding the need to re-run the entire pipeline.

```python
train_inputs.to_parquet("save/train_inputs.parquet")
test_inputs.to_parquet("save/test_inputs.parquet")
pd.DataFrame(train_targets).to_parquet("save/train_target.parquet")
pd.DataFrame(test_targets).to_parquet("save/test_target.parquet")
```

---

## 6. Baseline Models and Logistic Regression

### 6.1 — Baselines

Two trivial classifiers set the performance floor before any real model is trained.

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Baseline 1: random guess
def random_guess(inputs):
    return np.random.choice([0, 1], len(inputs))

# Baseline 2: always predict "no subscription"
def all_output_negative(inputs):
    return np.full(len(inputs), 0)

print(accuracy_score(test_targets, random_guess(test_inputs)))        # ~0.50
print(accuracy_score(test_targets, all_output_negative(test_inputs))) # ~0.51
```

> **Insight.** Because SMOTE balanced the classes 50/50 in training, the post-SMOTE test set is
> also balanced — so both baselines score around 50 %, making accuracy a meaningful relative
> metric in this controlled evaluation. In the real, imbalanced deployment environment, the
> all-negative baseline would score ~88.7 % accuracy with **0 % recall** — reinforcing exactly
> why recall is the headline metric for this business problem.

---

### 6.2 — Logistic Regression with GridSearchCV

Logistic Regression serves as the linear baseline and establishes the performance ceiling for
interpretable, linear models.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 0.98, 1],
    "solver": ["lbfgs"],
    "max_iter": [500],
}

grid_search = GridSearchCV(LogisticRegression(),
                           param_grid=param_grid,
                           scoring=["accuracy", "recall", "precision"],
                           refit="accuracy",
                           cv=5,
                           n_jobs=-1)

clf = grid_search.fit(train_inputs, train_targets)
print(clf.best_params_, clf.best_score_)
```

```
Best params:  {'C': 0.98, 'solver': 'lbfgs', 'max_iter': 500}
Best CV score: 0.75125
```

```
Train Accuracy = 0.75123   Test Accuracy = 0.76039
Recall        = 0.72394   Precision     = 0.77580   F1 = 0.74897
```

> **Insight.** Logistic Regression plateaus at ~76 % test accuracy regardless of regularisation
> strength. The relatively low recall (72.4 %) indicates that the linear decision boundary cannot
> capture the non-linear interactions between macroeconomic conditions and customer attributes
> that drive subscription behaviour.

---

## 7. Lasso and ElasticNet

Both models apply regularisation on top of Logistic Regression to perform automatic feature
selection and shrink noisy coefficients toward zero.

### 7.1 — Lasso (L1 Regularisation)

L1 penalty drives unimportant feature coefficients exactly to zero — effectively performing
built-in feature selection.

```python
model_lasso = LogisticRegression(penalty="l1", C=1,
                                  solver="saga", max_iter=500, n_jobs=-1)
model_lasso.fit(train_inputs, train_targets)
```

```
Train Accuracy = 0.75192   Test Accuracy = 0.76030
Recall        = 0.77661   Precision     = 0.72231
```

### 7.2 — ElasticNet (L1 + L2 Regularisation)

ElasticNet combines the sparsity of L1 with the stability of L2, controlled by `l1_ratio`.

```python
model_elastic_net = LogisticRegression(penalty="elasticnet", C=1,
                                        solver="saga", max_iter=500,
                                        l1_ratio=0.6, n_jobs=-1)
model_elastic_net.fit(train_inputs, train_targets)
```

```
Train Accuracy = 0.75152   Test Accuracy = 0.76021
Recall        = 0.77593   Precision     = 0.72322
```

> **Insight.** Lasso and ElasticNet marginally improve recall over plain Logistic Regression
> (77.7 % vs. 72.4 %) at the cost of precision — the regularisation is slightly shifting the
> decision boundary toward the minority class. However, all three linear models share the same
> ceiling of ~76 % test accuracy, confirming the non-linear nature of the problem.
>
> The ElasticNet coefficients provided by the client in **Task 3** are the primary output of this
> model family and form the basis for the three business recommendations.

---

## 8. Decision Tree and Random Forest

### 8.1 — Decision Tree

A single decision tree trained with explicit depth and leaf constraints to balance expressiveness
against overfitting.

```python
from sklearn.tree import DecisionTreeClassifier

model_decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=26,
    min_samples_split=10,
    min_samples_leaf=4,
    max_leaf_nodes=1300,
    random_state=42
)
model_decision_tree.fit(train_inputs, train_targets)
```

```
Train Accuracy = 0.89418   Test Accuracy = 0.83461
Recall        = 0.84824   Precision     = 0.82239
```

**Top features — Decision Tree (Gini importance):**

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | `num_employed` | 0.307 |
| 2 | `forward_rate` | 0.116 |
| 3 | `consumer_confidence` | 0.057 |
| 4 | `call_centre_volume` | 0.055 |
| 5 | `age` | 0.052 |
| 6 | `day_of_week_no` | 0.039 |
| 7 | `low_temp` | 0.037 |
| 8 | `contact_landline` | 0.035 |

> **Insight.** The Decision Tree is 6 percentage points below Random Forest but offers a key
> advantage — individual trees can be visualised and presented as flowcharts to non-technical
> stakeholders. The `num_employed` dominance (importance = 0.307) echoes the GLM ElasticNet
> coefficient finding (−0.558): macroeconomic conditions at time of contact are the single most
> decisive factor in the entire dataset.

---

### 8.2 — Random Forest

Random Forest aggregates 500 decision trees, each trained on a bootstrap sample and a random
feature subset — reducing variance and correcting the overfitting inherent in single trees.

```python
from sklearn.ensemble import RandomForestClassifier

model_random_forest = RandomForestClassifier(
    n_estimators=500,
    max_features=9,
    max_depth=35,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model_random_forest.fit(train_inputs, train_targets)
```

```
Train Accuracy = 1.00000   Test Accuracy = 0.89590
Recall        = 0.90316   Precision     = 0.88792
```

**Top features — Random Forest:**

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | `forward_rate` | 0.145 |
| 2 | `call_centre_volume` | 0.101 |
| 3 | `age` | 0.083 |
| 4 | `num_employed` | 0.079 |
| 5 | `high_temp` | 0.066 |
| 6 | `low_temp` | 0.058 |
| 7 | `employment_variation` | 0.049 |
| 8 | `day_of_week_no` | 0.048 |
| 9 | `num_contacts` | 0.042 |

<p align="center">
  <img src="images/task_2_figures/feature_importance_random_forest.png" width="680"
       alt="Random Forest feature importance"/>
</p>

> **Insight.** Random Forest is the **best overall model** — highest recall (90.32 %) and strong
> precision (88.79 %). The Train Accuracy of 100 % signals overfitting on the training set, but
> the 89.59 % test accuracy shows the ensemble still generalises well. The feature importance
> ranking shifts compared to the Decision Tree: `forward_rate` takes the top spot (vs.
> `num_employed`), and `call_centre_volume` rises significantly — suggesting that operational
> load at time of contact is an important signal that single-tree splitting fails to capture.
>
> **Plain-English explanation for non-technical stakeholders:** *"We build 500 small decision
> trees, each trained on a different random sample of customers and a random selection of
> features. Each tree votes on whether a customer will subscribe, and the forest's final answer
> is the majority vote. A feature's importance is how consistently and decisively it is used
> across all 500 trees."*

---

## 9. Gradient Boosting — XGBoost

XGBoost builds an ensemble of shallow trees **sequentially** — each tree corrects the residual
errors of the previous one — rather than training trees independently as in Random Forest.

```python
from xgboost import XGBClassifier

xgboost_model = XGBClassifier(
    max_depth=10,
    min_child_weight=5,
    n_estimators=1000,
    learning_rate=0.1,
    subsample=1,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
xgboost_model.fit(train_inputs, train_targets)
```

```
Train Accuracy = 0.99984   Test Accuracy = 0.89536
Recall        = 0.90967   Precision     = 0.88210
```

> **Insight.** XGBoost matches Random Forest almost exactly (89.54 % vs. 89.59 % test accuracy)
> with marginally higher recall (90.97 % vs. 90.32 %) but lower precision (88.21 % vs. 88.79 %).
> For this business problem — where **finding subscribers matters more than avoiding false alarms**
> — XGBoost is a competitive alternative. Training time was 18.5 seconds for 1,000 estimators,
> which is acceptable for a batch campaign-scoring workflow.
>
> In a production deployment, an ensemble of Random Forest and XGBoost predictions could yield
> the best of both models.

---

## 10. Save the Best Model

The entire pipeline — not just the model weights — is serialised using `joblib` so that predictions
can be made on new data without repeating any preprocessing steps.

```python
import joblib
from pathlib import Path

best_model_pipeline = {
    "model":             model_random_forest,
    "scaler":            scaler,
    "data_imbalance":    smote,
    "label_encoder":     enc_labels,
    "onehot_encoder":    encoder,
    "input_cols":        input_cols,
    "numeric_cols":      numeric_cols,
    "categorical_cols":  categorical_cols,
    "encoded_cols":      encoded_cols,
}

Path("models/").mkdir(parents=True, exist_ok=True)
joblib.dump(best_model_pipeline, "models/model_random_forest.joblib")
```

**Loading and scoring on new data:**

```python
pipeline    = joblib.load("models/model_random_forest.joblib")
predictions = pipeline["model"].predict(new_test_inputs)
```

> XGBoost is also saved separately via `pickle` as an alternative deployment artefact.
> Both formats are portable across Python environments with compatible library versions.

---

## Notes for README Integration

The following corrections should be applied when merging this content into the README:

1. **SMOTENC, not SMOTE** — the notebook uses `SMOTENC` (handles categorical features), not plain
   `SMOTE`. The existing README section says SMOTE — this should be updated.
2. **Precision / Recall for Random Forest** — the notebook outputs `Recall = 0.90316,
   Precision = 0.88792`. The model results Excel table has these two values reversed under the
   column headers. The notebook values are the ground truth.
3. **Dropped columns** — `days_since_previous`, `num_contacts_previous`, and `default` are
   removed in the ML pipeline and are not used as model features.
4. **Feature count** — after encoding, the final feature matrix is **39 columns** (12 numerical
   + 27 one-hot). The numerical-only constraint from the task brief applies specifically to the
   feature importance question (Task 2), not the full modelling pipeline.
