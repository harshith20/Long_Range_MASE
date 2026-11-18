# E06 — Open-set label audit (50 random wrong answers)

2026-11-18 · local, no Kaggle run · input: `correctness/wrong.csv` of kernel v5 (E05) · sample seed 0
Data: [`audit_50.csv`](audit_50.csv) — `claude_note` = first reading, `your_label` = user's decision (pending).

## Why
Open-set relations have only 5.1% accuracy (E04/E05). Before adding datasets, check whether some "wrong"
open-set answers are actually correct (matching misses) — and get a measured label error rate for the thesis.

## Setup
50 random rows from the 8,849 open-set `wrong` answers (all 9 open-set relations), with a surname/word-overlap hint.

## Results (first reading; final after the user's labels)
| Reading | n |
|---|---|
| wrong | 44 (14 of them: short title shared by several works) |
| wrong but subject-linked (AH candidates) | 2 |
| refusal not caught ("Not specified") | 2 |
| likely correct (answer more specific than gold: Seoul vs "South Korea") | 1 |
| garbled output | 1 |

## Insights
- Matching false negatives are rare (~1 in 50, ~2%): the 5% open-set accuracy is real model ignorance, so
  better matching will not create many open-set FA → more data (TriviaQA, EntityQuestions pilot) is the fix.
- "Not specified" is a refusal that the pattern list misses (~4% of this sample).
- ~28% of the sample are short titles shared by several works (*Angel*, *Fire*, *Boom*): the answer may be right
  for a different work. This is the ambiguity risk for AH labels — the M3 ambiguity check must handle it.

## Next steps
1. User confirms or corrects `your_label`; record decisions in `overrides/` if any row changes label.
2. Add "not specified" to `configs/refusal_patterns.yaml` (with the next correctness rerun).
3. Keep the M3 ambiguity check on AH rows (§3) — this sample shows it is needed.
