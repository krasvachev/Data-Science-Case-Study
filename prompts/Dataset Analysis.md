I have a tabular dataset for a binary/multi-class classification ML project (adjust as needed). I'm uploading a CSV file. Please perform a complete EDA and pre-modelling analysis following this framework:

Phase 1 — Data Ingestion & First Look Show shape, dtypes, head, and descriptive statistics. Flag any type mismatches or naming issues.

Phase 2 — Target Variable Analysis: Identify the target column, show class distribution, flag imbalance, and recommend an appropriate evaluation metric.

Phase 3 — Missing Value Analysis Count and visualise missing values per column. Classify missingness type where possible and recommend an imputation or dropping strategy.

Phase 4 — Univariate Analysis Plot distributions for numeric features (histograms, boxplots) and bar charts for categoricals. Flag skewed, low-variance, or high-cardinality columns.

Phase 5 — Bivariate Analysis (Features vs. Target) Show how each feature relates to the target using grouped plots and mutual information scores. Highlight the most and least predictive features.

Phase 6 — Multivariate Analysis & Correlations: Generate a correlation matrix and flag highly correlated pairs (>0.85). Note any redundant features.

Phase 7 — Outlier Detection: Apply IQR or Z-score method. Flag outliers and recommend whether to remove, cap, or keep them.

Phase 8 — Data Quality & Consistency Checks: Check for duplicate rows, inconsistent category labels, and potential target leakage.

Phase 9 — Pre-Modelling Preparation: Recommend and apply the following with code:

- Stratified train/val/test split
- Feature engineering opportunities
- Encoding strategy per categorical column
- Scaling strategy based on feature distributions
- Imbalance handling strategy (if needed)
- Feature selection based on importance and correlation

Phase 10 — Baseline: Implement a DummyClassifier and Logistic Regression baseline. Report performance using the recommended metric from Phase 2.

Output requirements:

- Provide a table that overviews the most important details of the dataset
- After each phase, include a short "Key Findings" summary
- At the end, provide an "Analysis Summary & Recommendations" section with a prioritised list of actions before modelling
- Use pandas, numpy, matplotlib, seaborn, and scikit-learn

The dataset: Check the uploaded CSV file.

Target column name: "outcome"  

Important Note: Use a high reasoning effort.