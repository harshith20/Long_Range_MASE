# E16 — M4 probes on all PopQA (SEPs, correctness baseline, familiarity)

2026-11-22 · commit 4b57a09 · stages correctness,probes · **GPU off** · kernel v12 · 107 min (probes 106 min,
1,575 fits) · **STATUS ⚠️ WARN** (`capital of` refusals, known; seq_nll probe pooled − within gap 0.12) ·
Report: [`reports/2026-11-22_qwen2.5-3b_popqa_full_correctness+probes_v12.md`](../../reports/2026-11-22_qwen2.5-3b_popqa_full_correctness+probes_v12.md)

(Numbered E16 because E15 is reserved for the M3 manual check, which is still being filled.)

## Why
M4 (§8 step 7): can single-pass linear probes replace the sampled entropies (H1b)? First look at H2
(entropy on AH vs FA) and H4 (subject familiarity separates UH, not AH), using the final M3 labels.

## Setup
New `probes` stage. Logistic regression on standardized hidden states with balanced class weights, using
nested CV grouped by subject. Per outer fold, the layer (indices 4–36), the position (TBG/SLT) and C ∈ {0.001,
0.01, 0.1, 1} are chosen on an inner 20% subject split of the training folds. SEP targets are binarized per fold
at the SEP best split. Familiarity target: s_pop > training median, from the subject-token state.
Labels are unchanged (2,128 / 11,114 / 939 / 86).

## Results (within-relation macro AUROC, 13 relations; pooled in brackets)
| score | target AUROC | correct vs wrong | FA vs UH | FA vs AH (7 rel.) | AH vs UH (7 rel.) |
|---|---|---|---|---|---|
| **SEP sem_entropy** (SLT @28) | 0.952 | **0.827** (0.910) | 0.860 | 0.646 | 0.732 |
| SEP seq_nll (TBG @32–36) | 0.935 | 0.766 (0.889) | 0.790 | 0.646 | 0.632 |
| correctness probe (SLT @24) | 0.955 | **0.893** (0.955) | 0.914 | **0.800** | 0.653 |
| familiarity (subject @20) | 0.822 | 0.714 (0.587) | 0.741 | 0.592 | 0.648 |
| sampled sem_entropy (ref.) | – | 0.781 (0.898) | 0.807 | 0.648 | 0.683 |
| greedy tok_max (ref.) | – | 0.797 (0.906) | 0.824 | 0.653 | 0.696 |

Mean P(popular) from the familiarity probe: FA 0.578 · **AH 0.579** · UH 0.476 (Spearman with s_pop 0.68).

## Insights
- **H1b holds, and the probe is better than sampling.** The single-pass SE probe reaches 0.827 within-relation,
  vs 0.781 for sampled SE (10 samples + NLI) and 0.797 for greedy tok_max. SLT (an answer token) beats TBG at
  every layer; the best layers are mid-late (20–28), not the last.
- **The direct correctness probe is the strongest single score** (0.893 within). M5 must beat it, or say
  plainly that it did not (§3 Baselines).
- **H2 is partly supported.** Entropy separates FA vs AH much worse (0.65) than FA vs UH (0.81–0.86), but not at
  chance. In open-set relations the SE probe reaches only 0.54 on FA vs AH (closed 0.70). AH is more uncertain
  than FA mostly where the answer set is small.
- **H3 hint:** the correctness probe separates FA vs AH at 0.80, so the hidden state holds a signal beyond
  entropy. The M5 ablations should test which features recover it.
- **H4 is supported clearly.** Familiarity is identical for FA and AH (0.578 vs 0.579; FA vs AH pooled
  0.516) and lower for UH. So AH = familiar subject + wrong fact.
- **Familiarity is the reverse of the relation confound:** within-relation 0.714 > pooled 0.587. Accuracy is
  U-shaped in s_pop (closed-set relations hold many obscure subjects), so the relations mask the effect when
  pooled.
- The pooled − within gap is 0.06–0.12 for every entropy score again (H5). Only within-relation numbers go in the thesis tables.
- Threat: AH vs FA has only 7 usable relations and 740 AH. TriviaQA / EntityQuestions are needed for H8 and for more open-set AH.

## Next steps
1. **E15 manual check** (user filling `overrides/m3_manual_popqa_qwen2.5-3b.csv`) → AH precision/recall → M3 done.
2. **M5 classifier** on `probes/oof_scores.jsonl` + M2 single-pass features. Feature ablations: entropy only →
   + layer-wise → + attention → + familiarity, vs the correctness-probe and relation-only baselines; two-stage
   and flat. Main question: does anything beat 0.80 on FA vs AH?
3. Optional M4 follow-up: an AH-vs-FA probe directly on hidden states (an H3 upper bound). It needs care,
   because labels would become a probe target. Use it as a baseline only, like the correctness probe.
4. TriviaQA generation + EntityQuestions pilot on Kaggle GPU (needs your explicit go for `generation`).
5. Runtime: 106 min on CPU; M5 reruns should load `oof_scores.jsonl`, not refit the probes.
