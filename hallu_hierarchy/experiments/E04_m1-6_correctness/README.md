# E04 — M1.6: correctness rules on all PopQA

2026-11-18 · commit 0155396 · stage correctness · GPU off · kernel v4 · ~2 min · **STATUS ✅ PASS**
Report: [`reports/2026-11-18_qwen2.5-3b_popqa_full_correctness_v4.md`](../../reports/2026-11-18_qwen2.5-3b_popqa_full_correctness_v4.md)
Attempt 1 (kernel v3) crashed in notebook cell 1 — script bug (`false` written into Python), fixed in 0155396.

## Why
Apply the M1.6 rules (§3) to all saved PopQA generations: get labels that are clean enough for M3 (AH/UH) and
quantify the relation confounder by open/closed group. First run through the Kaggle CLI.

## Setup
vs E03 labels: head-word stripping of gold aliases ("comedy film" → "comedy"), relaxed-only matches → `partial`
(excluded from training), "none" is not a refusal for `religion`, ambiguity flag is a column only.

## Results
| | Count | Share | vs E03 |
|---|---|---|---|
| correct | 2,047 | 14.3% | +105 (head words) |
| wrong | 11,268 | 79.0% | −264 |
| abstain | 787 | 5.5% | −6 (religion "None") |
| partial | 165 | 1.2% | new |

| Group | n | Accuracy | Partial |
|---|---|---|---|
| closed-set | 4,384 | 35.3% | 150 |
| open-set | 9,883 | 5.1% | 15 |

Accuracy by split: train 14.3% · val 14.5% · test 14.4%. Excluded from training: 953.

## Insights
- The new rules work and pass all checks; `partial` is small (1.2%), so no manual review is needed.
- **92 of 165 partials are religion** ("Catholic" vs gold "catholic church", "Muslim" vs "muslim religion").
  These are correct answers in substance → `church` and `religion` added as head words (E05).
- **Confounder confirmed and large:** closed-set relations give 1,546 of 2,047 correct answers (76%) from 31% of
  questions. Pooled AUROC will be inflated; within-relation results are essential (H5, rule 11).
- **Open-set correct answers are scarce:** 501 in total; e.g. screenwriter 31, producer 25, director 49. With a
  20% test split, most open-set relations will have < 30 correct test examples → within-relation AUROC will be
  limited to a few relations unless TriviaQA (more correct answers about people) is added.
- Abstain is high for `capital of` (32.8%) and `producer` (12.3%) — not yet checked whether these are true
  refusals or real answers caught by the "is not" pattern.
- Splits are balanced (14.3–14.5%), consistent with the subject-based split being sound.

## Next steps
1. E05: rerun correctness with the two new head words (CPU, minutes) — expected religion partials → correct.
2. Check abstain per relation (`capital of`, `producer`) — the abstain rule must not hide real answers.
3. Subject-state stage (H4, GPU) → M2 on a 200-question subset.
4. Plan within-relation evaluation now: count correct/wrong per relation in the test split before M5; bring
   TriviaQA forward if open-set counts are too small.
