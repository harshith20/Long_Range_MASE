# E07 — M1.6c: 5-fold CV labels + "not specified" refusals

2026-11-18 · commit 6de0783 · stage correctness · GPU off · kernel v6 · ~2 min · **STATUS ⚠️ WARN** (expected)
Report: [`reports/2026-11-18_qwen2.5-3b_popqa_full_correctness_v6.md`](../../reports/2026-11-18_qwen2.5-3b_popqa_full_correctness_v6.md) · follows [E05](../E05_m1-6b_correctness/), [E06](../E06_open-set_label_audit/)

## Why
Write the subject-grouped `fold` column into the labels (evaluation protocol decided after E05) and treat
"Not specified" / "Not available" as refusals (found in the E06 audit), so M2 and M3 start from final labels.

## Setup
vs E05: 5-fold CV assignment by subject; exact refusal patterns `not specified`, `unspecified`, `not available`.

## Results
| | E05 | E07 |
|---|---|---|
| correct | 2,128 (14.9%) | 2,128 (14.9%) |
| wrong | 11,266 | **11,114** |
| abstain | 787 (5.5%) | **939 (6.6%)** — +145 "not specified", +7 "not available" |
| partial | 86 | 86 |
| relations usable within-relation (≥ 30 correct and wrong) | 5 of 16 (test split) | **13 of 16** (5-fold CV) |

Folds: 2,846 / 2,889 / 2,824 / 2,851 / 2,857 questions, no subject in two folds.
Not usable: producer (25 correct), mother (15), color (10 wrong). Open-set usable: director, screenwriter,
author, composer, place of birth, father, occupation.

## Insights
- CV fixes the evaluation-size problem without new data: open-set within-relation AUROC is now possible for
  7 of 9 open-set relations (smallest: screenwriter, 31 correct).
- 152 refusals no longer pollute the `wrong` pool that M3 turns into AH/UH; in the samples, 671 "not specified"
  now form refusal clusters in M2 instead of fake answers.
- The `capital of` WARN (34% abstain) is stable and explained (premise denials, "Not applicable"); it will
  repeat on every correctness run unless the threshold is raised for that relation.
- Labels for PopQA are now final for M2/M3 (barring the open `capital of` premise-denial decision).

## Next steps
1. Subject-state stage (H4, GPU) — prompt-only forward pass, subject-token states.
2. M2 on a 200-question subset (entropy features, NLI clustering), then on all.
3. Decide whether `capital of` premise denials stay `abstain` (before M3).
