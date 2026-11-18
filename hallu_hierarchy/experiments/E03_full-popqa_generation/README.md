# E03 — Full PopQA generation (14,267 questions)

2026-11-17 · commit 2ab927f · stages data, generation, correctness · GPU T4 (browser) · ~2.9 h
Outputs: Kaggle Dataset `harshithswamy20/hallu-hierarchy-m-1-5` (2.4 GB: generations, hidden states, attention).
Report not kept (pre-dates `reports/`).

## Why
Produce the full set of answers, samples and internal states once, so every later stage (correctness, subject
states, M2–M5) runs from saved outputs without regenerating.

## Setup
All of PopQA, checkpoints every 100 questions; generation now also saves per-head attention of the last prompt
token (subject mass per layer × head, full rows at 3 middle/late layers) for M2.

## Results
| | Value |
|---|---|
| Finished | 14,267 / 14,267, 143/143 checkpoints, no empty answers, subject located in 100% of prompts |
| Speed | 0.74 s per question (~3× slower than estimated from M1) |
| Correct (strict, M1.5 rules) | 1,942 (13.6%); wrong 80.8%; abstain 5.6% |
| `ambiguous_subject` | 48.8% (~5,800 rows would have been set aside) |
| Strict/relaxed disagreements | 261 — too many to review by hand |

## Insights
- **Relation confounder:** ~74% of correct answers come from closed-set relations, which are ~31% of questions.
  A detector could score well by recognising the question type — within-relation evaluation is required (H5).
- Correct answers are scarce (~1.9k of 14k), especially for open-set relations.
- Time per question is the real budget constraint for TriviaQA and OLMo runs (plan ~3 h per 14k on a T4).

## Next steps
1. M1.6: `partial` label instead of manual review, head-word stripping, ambiguity flag as column only (E04).
2. Kaggle CLI automation so runs are reproducible and logged (§12).
