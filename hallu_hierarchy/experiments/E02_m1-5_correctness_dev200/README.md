# E02 — M1.5: correctness fixes on the 200

2026-11-17 · commits 9377dbc (first run), 2ab927f (final) · stage correctness · GPU off (browser) · ~1 s
Inputs: E01 outputs. Final report numbers below (reports not kept, pre-date `reports/`).

## Why
Test the M1.5 correctness rules on saved generations (no GPU): separate refusals, measure what relaxed matching
would add, apply manual overrides, flag ambiguous subjects.

## Setup
Refusal patterns, strict vs relaxed matching, generic-alias cleaning ("church" …), 3 manual overrides
(Avatar → correct, Edmund A. Walsh → correct, Atlanta → gold_noise), single-word `ambiguous_subject` heuristic.

## Results
| | First run (9377dbc) | Final (2ab927f) |
|---|---|---|
| Correct | 34 (17.0%) | 36 (18.0%) |
| Wrong | 153 | 151 |
| Abstain | 13 (6.5%) | 13 (6.5%) |
| Relaxed would add | +3 | — (3 decided by override) |
| `ambiguous_subject` flags | 101 (50.5%) | 99 (49.5%) → STATUS WARN |

## Insights
- 13 refusals were previously counted as hallucinations — abstain handling matters for clean AH/UH labels.
- Relaxed matching adds very little (+3 / 200); manual review is feasible here but not at 14k scale.
- The single-word ambiguity heuristic is far too broad (~50%); using it to exclude rows would discard half the data.
- Creative-work relations (screenwriter 3%, producer 4%, director 0%, composer 0%) are near zero accuracy;
  capital 64%, country 38% — first sign of the relation confounder.

## Next steps
1. Generate all of PopQA (E03) — 200 questions are too few for any classifier.
2. Replace manual review by a rule (`partial`) and make the ambiguity flag a column only (done in M1.6, E04).
