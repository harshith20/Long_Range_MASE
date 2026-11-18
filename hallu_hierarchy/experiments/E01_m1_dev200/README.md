# E01 — M1: first end-to-end run (200 PopQA questions)

2026-11-16 · commit 903a6e0 · stages data, generate, correctness · GPU T4 (browser run) · ~40 s generation
Outputs: Kaggle Dataset `harshithswamy20/hallu-m1-exp1`. Report not kept (pre-dates `reports/`).

## Why
Check that data loading, subject-based splits, generation with hidden-state/attention caching, checkpointing and
correctness labeling work on Kaggle before scaling up (M1).

## Setup
Qwen2.5-3B-Instruct, 200 random PopQA questions (one per subject), greedy + 10 samples, alias matching
(`contains` mode, refusals not handled).

## Results
| | Value |
|---|---|
| Finished | 200 / 200, all greedy answers ended with EOS, ~4 tokens per answer |
| Correct | 34 (17%) |
| Leakage | none (one question per subject) |

## Insights
- The pipeline works; generation is fast and answers are short, so `max_new_tokens=32` is ample.
- Correctness had three problems: strict alias matching misses some correct short forms; some questions have
  ambiguous titles or wrong gold answers; ~7% of answers are refusals but were counted as hallucinations.

## Next steps
1. M1.5: refusal → `abstain`, strict vs relaxed matching, overrides, flags — fixes the three problems above.
