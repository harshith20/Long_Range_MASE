# E19 — Four tests on the AH mechanism (local)

2026-11-23 · code at commit c3e6747 · local CPU · scripts: `t23_assoc_type.py` (tests 2, 3; Wikipedia cache) and
`t14_probes.py <path to hidden_slt.npy>` (tests 1, 4; answer-token states from dataset `hallu-hierarchy-m-1-5`), run
from the repo root with `PYTHONPATH=.` · ~25 min including the 0.6 GB download · no report file (numbers below)

## Why
E18 showed that the correctness probe separates FA vs AH (0.80) partly through answer identity. Is there a real,
transferable "this answer is wrong for this subject" signal? What makes an AH (association, type, role)? (H2, H3, H9)

## Results (within-relation AUROC unless noted)
| test | result |
|---|---|
| 1. correctness probe closed → open | correct vs wrong 0.828 (in-group 0.896) · **FA vs AH 0.593** (in-group 0.746) |
| 1. correctness probe open → closed | correct vs wrong 0.770 (in-group 0.892) · FA vs AH 0.743 (in-group 0.862) |
| 4. probe trained directly on FA vs AH (SLT, layers 20–28) | **0.824** (pooled 0.887; open 0.796, closed 0.908; father 0.65, place of birth 0.70) |
| 4. … same answer string (FA and AH with identical answers) | **0.545** (correctness probe: 0.61) |
| 2. gold-answer mentions on the subject page (median) | FA 11 · AH 5 · UH 5 → FA vs AH 0.686, FA vs UH 0.731 |
| 2. AH answer mentioned more often than the gold | 38% of AH (open 38%, closed 39%) |
| 3. answer is a valid value of the asked relation | FA 93% · AH 47% · UH 36% |
| 3. answer is only a value of another relation (role swap) | AH 12% (capital of → country 28, place of birth → capital 21, producer → director 4) · UH 7% |

## Insights
- **The hidden-state FA vs AH signal is mostly answer identity, not a "this fact is wrong" signal.** A dedicated
  FA-vs-AH probe reaches 0.82, but only 0.55 when FA and AH give the same answer string. It barely transfers across
  relation groups (closed → open 0.59). What transfers is a generic correct-vs-wrong direction (0.77–0.83).
- So AH is hard **even for hidden-state probes** once you control for the answer: H2 holds beyond output entropy.
- **AH happens on less salient facts:** the gold is mentioned half as often on the subject's page for AH (5) as
  for FA (11). Gold salience alone separates FA vs AH (0.686) as well as entropy does (0.684): a fact-frequency
  effect, the H9 mechanism, measurable now with a Wikipedia proxy.
- **Overshadowing is real but not the majority:** in 38% of AH the wrong answer is mentioned more than the gold.
  Typical patterns are containment / role swaps: county → state (capital of → country), small town → capital city
  (place of birth → capital), director given as producer.
- AH answers are mostly the right *type* (47% valid for the relation vs 36% of UH), so type checking alone won't catch them.
- Threat: religion → mother (20) is likely an alias artefact (same strings as gold values elsewhere); the
  Wikipedia page proxy is a weak stand-in for pretraining frequency (M6 replaces it with infini-gram).

## Next steps
1. M6: infini-gram counts of subject + gold vs subject + answer on OLMo's corpus. This is the real test of
   "AH = less frequent fact, overshadowed by a more frequent association" (H9).
2. Add the gold-salience proxy as an *analysis* variable (never a feature: it uses the gold answer).
3. Report the FA-vs-AH probes with a same-answer control. Without it, hidden-state AUROCs overstate what the model "knows".
