# Task 2 - Machine Learning · Concise Interview Solution

This folder contains the **concise solution** for **Task 2 - Machine Learning**. It is the version you should actually use during the interview.

## Files

| File | Purpose |
|------|---------|
| `LittleBank_Case_Study_ML_Concise_Solution.ipynb` | Jupyter notebook - run interactively |
| `littlebank_case_study_ml_concise_solution.py` | Script equivalent - run from the command line |

## Why "Concise"?

The detailed solution in the repository root (`LittleBank_Case_Study_ML.ipynb`) trains six model families and includes exhaustive hyperparameter tuning. This concise version focuses on the essentials you can realistically reproduce within the 3-hour window:

- Data cleaning tailored for ML.
- Class-imbalance handling with **SMOTE**.
- Preprocessing with `MinMaxScaler` + `OneHotEncoder`.
- A compact model comparison: Baseline → Logistic Regression → Decision Trees → Random Forest → XGBoost.
- **Feature importance** plots.
- Evaluation on **Recall** (primary), Precision, and Accuracy.

## Key Results (from this pipeline)

| Model | Test Accuracy | Recall | Precision |
|-------|:-------------:|:------:|:---------:|
| Logistic Regression | 0.7604 | 0.7239 | 0.7758 | 
| **Random Forest** 🏆 | **0.8959** | **0.9032** | **0.8879** | 
| Extreme Gradient Boosting | 0.8954 | 0.9097 | 0.8821 |

> **Primary metric is recall** - we care about finding real subscribers, not maximising overall accuracy.

## Explain Each Model in One Sentence

The brief asks you to explain every algorithm to a non-technical client. Rehearse these until you can say them without notes.

| Technique | Plain-language explanation |
|-----------|----------------------------|
| **Baseline** | "The score you would get by guessing. Any real model has to beat it." |
| **Logistic Regression** | "Gives each customer trait a weight and adds them up into a probability of subscribing." |
| **Lasso / ElasticNet** | "The same model, but it pushes the weights of unimportant traits towards zero, so only the traits that matter remain." |
| **Decision Tree** | "A flowchart of yes/no questions, e.g. 'Called in May?', that ends in subscribe or not." |
| **Random Forest** | "Hundreds of different flowcharts, each built on a random sample of customers, that vote on the answer." |
| **XGBoost** | "Flowcharts built one after another, each one correcting the mistakes of the previous ones." |
| **SMOTE-NC** | "Only 11% of customers subscribed, so we create realistic synthetic subscribers to give the model enough examples to learn from." |
| **Feature importance** | "How much each trait helped the model separate subscribers from non-subscribers. It shows correlation, not cause." |
| **Recall** | "Of all the customers who would subscribe, the share the model finds." |

## How to Practise

1. **Open the blank exam** in Colab or Jupyter and read the *Rules* cell. The setup cell finds the data file and installs `imbalanced-learn` if it is missing.
2. **Start a 70-minute timer**, then run the first cell. Copy sections 0–1 from your Task 1 notebook where you can.
3. **Fill every `# TODO` cell, every ✍️ insight cell and the comparison table** in section 9. You can open a hint, but write down which ones you opened.
4. **Stop when the timer ends**, even if you haven't finished. Then restart the kernel and run all cells.
5. **Compare with the concise solution:**
   - Are your metrics within about ±0.01 of the table above? If not, look for differences in cleaning or splitting.
   - Did you beat both baselines, and can you say by how much?
   - Does your feature importance insight explain what the model means for the business, not just which features rank highest?
6. **Score yourself** with page 1 of the [Self-Check](../../practice_kit/Data_Science_Case_Study_Self_Check.md).
7. **Repeat** until you finish within 70 minutes without opening a hint.

> **Tip.** Prepare a one-liner for each model before the interview. For Random Forest: *"A committee of decision trees, each trained on a random subset of the data - the forest predicts by majority vote."*

## Related

- [← Back to main README](../../README.md)
- [← Task 1 · EDA blank exam and solution](../task_1_eda/README.md)
- [→ Task 3 · Business strategy and recommendations](../task_3_recommendations/business_strategy_and_recommendations.md)
- [Self-Check rubric](../../practice_kit/Data_Science_Case_Study_Self_Check.md)
- [Detailed ML solution](../../LittleBank_Case_Study_ML.ipynb) (six model families, full hyperparameter tuning)