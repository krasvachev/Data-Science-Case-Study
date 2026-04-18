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
