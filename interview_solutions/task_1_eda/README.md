# Task 1 — EDA · Concise Interview Solution

This folder contains the **concise solution** for **Task 1: Exploratory Data Analysis** of the LittleBank Case Study. It is the version you should actually use during a timed mock interview.

## Files

| File | Purpose |
|------|---------|
| `LittleBank_Case_Study_Concise_Solution.ipynb` | Jupyter notebook — run interactively |
| `littlebank_case_study_concise_solution.py` | Script equivalent — run from the command line |

## Why "Concise"?

The detailed solution in the repository root (`LittleBank_Case_Study.ipynb`) is ~140 cells and is designed for deep learning, not time-pressured interviews. This concise version strips the analysis down to the essentials you can realistically produce within the 3-hour window:

- Data overview (`df.info()`, `df.describe()`, class imbalance).
- Targeted data cleaning (duplicates, `"unknown"` categories, `-1` sentinels).
- 5–7 high-signal visualisations:
  1. Target class distribution.
  2. Success rate by month.
  3. Success rate by day of week.
  4. Success rate by profession.
  5. Age distribution split by outcome.
  6. Correlation heatmap of numerical features.
  7. Campaign contact analysis.
- A one-sentence business insight at the end of each section.

## How to Use for Interview Practice

1. Open the notebook and **clear all outputs**.
2. Set a **90-minute timer** (the recommended Task 1 budget).
3. Work top-to-bottom without looking at the solution.
4. At the end of the timer, compare your output to the reference notebook.
5. Repeat until you can complete it confidently within the time limit.

> **Tip.** After each section, practise saying your business insight out loud as if presenting to the head of loan sales. Big Four interviewers score you on **narrative**, not just code.

## Related

- [← Back to main README](../../README.md)
- [→ Task 2 · Machine Learning concise solution](../task_2_machine_learning/README.md)
