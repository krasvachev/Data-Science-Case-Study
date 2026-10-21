# Task 1 - EDA · Concise Blank Exam and Solution

> **What you do here:** solve the Task 1 blank exam against a 70-minute timer, then check it against the concise solution.
> **Time:** about 105 minutes: 70 for the exam, 5 to re-run the notebook, 30 to compare.

## Files

| File | What it is | When to open it |
|------|------------|-----------------|
| `LittleBank_Case_Study_Concise_Blank_Exam.ipynb` | [The blank exam](https://colab.research.google.com/github/krasvachev/Data-Science-Case-Study/blob/main/interview_solutions/task_1_eda/LittleBank_Case_Study_Concise_Blank_Exam.ipynb): `# TODO` code cells, hidden hints and ✍️ insight cells. No solutions. | **First.** Start the timer before you run the first cell. |
| `LittleBank_Case_Study_Concise_Solution.ipynb` | [The concise solution](https://colab.research.google.com/github/krasvachev/Data-Science-Case-Study/blob/main/interview_solutions/task_1_eda/LittleBank_Case_Study_Concise_Solution.ipynb), with outputs and insights. | **Only after the timer ends.** |
| `littlebank_case_study_concise_solution.py` | Script version of the solution. | Optional: run it from the command line. |

## Why "Concise"?

The detailed solution in the repository root (`LittleBank_Case_Study.ipynb`) is ~140 cells. It is designed for learning, not for time-pressured interviews. This concise version leads to the essentials, and you can re-produce within the 70-minute window:

- Data overview (`df.info()`, `df.describe()`, class imbalance).
- Targeted data cleaning (duplicates, `"unknown"` categories, `-1` sentinels).
- 5–7 high-signal visualisations
- A business insight at the end of each section.

## How to Practise

1. **Open the blank exam** in Colab or Jupyter and read the *Rules* cell. The setup cell finds the data file for you, so don't change the path.
2. **Start a 70-minute timer**, then run the first cell.
3. **Fill every `# TODO` cell and every ✍️ insight cell.** You can open a hint, but write down which ones you opened: each one is a topic to practise again.
4. **Stop when the timer ends**, even if you haven't finished. Then restart the kernel and run all cells. A notebook that doesn't run top to bottom isn't finished.
5. **Compare with the concise solution**, section by section:
   - Do your key numbers match (class balance, duplicates, monthly success rates)?
   - Did you find the same main result: **many calls, very few subscriptions**?
   - Does each of your insights include a number and an action?
6. **Score yourself** with page 1 of the [Self-Check](../../practice_kit/Data_Science_Case_Study_Self_Check.md).
7. **Repeat** until you finish within 70 minutes without opening a hint.

> **Tip.** After each section, practise saying your business insight out loud as if presenting to the head of loan sales. Big Four interviewers score you on **narrative**, not just code.

## Related

- [← Back to main README](../../README.md)
- [→ Task 2 · Machine Learning blank exam and solution](../task_2_machine_learning/README.md)
- [Self-Check rubric](../../practice_kit/Data_Science_Case_Study_Self_Check.md)
- [Detailed EDA solution](../../LittleBank_Case_Study.ipynb) (about 140 cells, for learning rather than timed practice)
