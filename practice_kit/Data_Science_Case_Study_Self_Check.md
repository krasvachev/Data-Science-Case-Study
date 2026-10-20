# Data Science Case Study — Self-Check

Use it twice: for 15 minutes **before you submit**, and for 30 minutes **before the interview**. 

---

## Page 1 — Before you submit

### Step 1. Deal-breakers (5 minutes)

Each of these can sink an otherwise good solution. If one is true, fix it now. If you can't, write it in your notebook as a known limitation, since an admitted flaw costs far less than a discovered one.

- [ ] I fitted something (scaler, encoder, resampling, feature selection, tuning) on data that includes the test set.
- [ ] My model uses information that would not exist at the moment the decision is made.
- [ ] I ignored an explicit instruction in the brief (columns to use, metric, audience, format).
- [ ] My headline metric looks good even for a useless model (e.g. accuracy when the target is rare).
- [ ] I say or imply that X *causes* Y, but I only showed they move together.
- [ ] I dropped rows or columns without checking how they relate to the target.
- [ ] A number in my text doesn't match my output. *Recompute your three most important numbers from raw data.*
- [ ] There is no conclusion or recommendation section.
- [ ] Something in the notebook would embarrass me if the client read it.

### Step 2. Score ten questions (10 minutes)

Score each question **0** (missing or wrong), **1** (partly), or **2** (clearly yes, and you can point to the cell). Be strict: "I did it in my head" scores 0.

| Hat | Question | What a 2 looks like |
| --- | --- | --- |
| Scientist | **1. Do I understand the data?** | What one row is; cleaning steps listed with counts and reasons; odd codes (`unknown`, `-1`, blanks) checked against the target; for each feature, when it becomes known |
| | **2. Does my exploration answer the question?** | Base rate first; segment rates shown with their counts; one-line "so what" under every chart; trend over time checked |
| | **3. Do I show how sure I am?** | Comparisons against a reference group; sample sizes stated; uncertainty shown (interval, test, or cross-validation spread) |
| | **4. Is my model a fair test?** *(skip if no modelling)* | Baseline and simple model first; split before any fitting; metric suited to the class balance; overfitting checked; complex model kept only if it clearly beats the simple one |
| Engineer | **5. Would it run for someone else?** | Runs top to bottom on a fresh kernel; seeds fixed; preprocessing inside a pipeline; no dead or duplicate cells |
| Consultant | **6. Did I answer the brief's actual question?** | The decision my work supports is named in the first paragraph, and every task is answered as asked |
| | **7. Are my insights specific and usable?** | 3–5 insights with numbers; the levers the client controls are separated from those it doesn't |
| | **8. Do my recommendations follow from the evidence?** | Each recommendation tied to a finding, with a confidence level, a limitation, and a way to test it (A/B or holdout) |
| | **9. Could a busy manager get it in 5 minutes?** | Summary at the top in plain language; charts titled with the finding |
| Interviewee | **10. Can I explain every choice in 30 seconds to a non-expert?** | You said it out loud for each major step without notes |

### Step 3. Read your score

Total out of 20, or out of 18 if you skipped question 4.

| Score | Meaning | Do this |
| --- | --- | --- |
| 80% or more, and no deal-breakers | Ready | Submit; spend your remaining time on page 2 |
| 60–79% | Fixable | Fix every 0 first, then the 1s in your weakest hat |
| Below 60%, or any deal-breaker left unmentioned | At risk | Fix deal-breakers, then questions 4, 6 and 8 |

**Weakest hat** = the group with the most 0s. That is where the interviewers will dig.

**Never trade a 2 for complexity.** A simple, validated model you can explain beats a complex one you can't defend.

---

## Page 2 — Before the interview

### Prepare one card per weakness (one line each)

For every deal-breaker and every 0, write: **"Yes, [what's wrong]. That means [consequence for the results]. I'd fix it by [action]."** Rehearse until each card takes under 30 seconds.

### Open with your weaknesses

In the first five minutes of your walkthrough, name your two or three biggest weaknesses and what you'd change. An admitted flaw reads as judgment; the same flaw found by the interviewer reads as a gap.

### Rehearse these questions out loud

1. Why this metric? What does a model that predicts the majority class score?
2. How much better is your model than a one-line rule?
3. In order, what did you fit, and on which rows?
4. Which feature would you *not* have at prediction time?
5. What was your most surprising result, and how did you check it?
6. What is this worth to the client, roughly?
7. What is the single biggest threat to your conclusion?
8. Does X cause Y? How would you find out?
9. What would you do with one more week?

### How you'll be judged in the discussion

- **The challenge is right:** agree in one sentence, then say what changes and what still holds.
- **The challenge is wrong:** disagree politely and use your own evidence. Interviewers often plant a confident wrong claim ("accuracy is fine here", "resampling the whole dataset is fine") to see if you fold.
- **You don't know:** say "I don't know; here's how I'd find out." Never bluff.
- **Structure:** answer first, then the reason. Keep it to 30 seconds unless they ask for more.
