# Task 1 - EDA · Concise Interview Solution

This folder contains the **concise solution** for **Task 1 - Exploratory Data Analysis**. It is the version you should actually use during the interview.

## Files

| File | Purpose |
|------|---------|
| `LittleBank_Case_Study_Concise_Solution.ipynb` | Jupyter notebook - run interactively |
| `littlebank_case_study_concise_solution.py` | Script equivalent - run from the command line |

## Why "Concise"?

The detailed solution in the repository root (`LittleBank_Case_Study.ipynb`) is ~140 cells. It is designed for learning, not for time-pressured interviews. This concise version leads to the essentials, and you can re-produce within the 1-hour window:

- Data overview (`df.info()`, `df.describe()`, class imbalance).
- Targeted data cleaning (duplicates, `"unknown"` categories, `-1` sentinels).
- 5–7 high-signal visualisations
- A business insight at the end of each section.

## How to Use for Interview Practice

1. Open the notebook and **clear all outputs**.
2. Set a **70-minute timer** (the recommended Task 1 budget).
3. Work top-to-bottom without looking at the solution.
4. At the end of the timer, compare your output to the reference notebook.
5. Repeat until you can complete it confidently within the time limit.

> **Tip.** After each section, practise saying your business insight out loud as if presenting to the head of loan sales. Big Four interviewers score you on **narrative**, not just code.

## Related

- [← Back to main README](../../README.md)
- [→ Task 2 · Machine Learning concise solution](../task_2_machine_learning/README.md)
