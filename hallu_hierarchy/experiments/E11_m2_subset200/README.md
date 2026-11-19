# E11 — M2 on a 200-question subset (entropy hierarchy, first real NLI run)

2026-11-19 · commit 513257a · stages correctness, entropy · GPU T4 · kernel v10 · 0 min queued + 3 min run ·
**STATUS ⚠️ WARN** (expected) · Report: [`reports/2026-11-19_qwen2.5-3b_popqa_full_correctness+entropy_v10.md`](../../reports/2026-11-19_qwen2.5-3b_popqa_full_correctness+entropy_v10.md)
Attempt 1 (kernel v9) crashed: DeBERTa loaded in fp16 → "expected scalar type Float but found Half"; fixed in 513257a.

## Why
Check every M2 feature on real data with the real DeBERTa-MNLI before running all 14,267 questions (§8 step 4),
and compare discrete vs probability-weighted semantic entropy (user decision: compute both).

## Setup
Random seeded 200 questions (36 correct, 151 wrong, 13 abstain); labels regenerated in the same run (= E09).

## Results
| Feature (AUROC wrong vs correct) | AUROC |
|---|---|
| sem_entropy (discrete) | **0.876** |
| seq_nll · tok_max · sem_entropy_w · lens_H_36 | 0.865 · 0.864 · 0.863 · 0.861 |
| tok_mean | 0.845 |
| attention to subject, block 32 (max / mean) | 0.169 / 0.174 → **higher for correct** (≈ 0.83 flipped) |
| l* · lens_H_18/24/30 | 0.64 · 0.58 / 0.40 / 0.43 |

Checks: lens = real output (99.77%), NLI sanity 3/3 (incl. "Paris" ⇒ "The city of Paris"), no NaN, labels from this run.
Mixed precision overflowed in DeBERTa (`c10::Half` overflow) → automatic fp32 fallback: 8,608 NLI pairs in 110 s.
Semantic clusters: mean 6.16 per question, 13% single-cluster — almost the same as exact string matching (6.28, 12% in E10).
Correlation of the final-layer levels 0.79–0.95; VIF: sem_entropy 28.5, sem_entropy_w 15.5, seq_nll 11.2.

## Insights
- The pipeline works end to end on Kaggle; all hard checks pass. Both fallbacks did their job.
- **DeBERTa barely merges answers on PopQA**: short entity answers ("Lyon", "Nice") are rarely paraphrases, so
  semantic entropy ≈ entropy over distinct strings. The expensive NLI adds little here (may differ on TriviaQA).
- **All final-layer uncertainty measures carry the same information** (AUROC 0.85–0.88, VIF up to 28): the
  "hierarchy" is close to one dimension on this data — rule 4 matters; logistic regression must not get all of them raw.
- Discrete SE is slightly better than weighted (0.876 vs 0.863); 36 correct answers are too few to call it.
- **Attention to the subject is the strongest non-entropy signal** (correct answers attend more), relevant for H3/H4 —
  but subject attention likely differs by relation type, so it must be checked **within relation** (H5).
- Mid-layer lens entropy alone is near chance; l* is weak.
- All AUROCs here are pooled over relations on 200 questions — inflated by the relation confounder until the
  within-relation numbers exist (none computable on 200).

## Next steps
1. M2 on all 14,267 questions — gives within-relation AUROCs for 13 relations. NLI in fp32 ≈ 614k pairs ≈ 2.2 h
   on a T4 unless sped up (length-sorted batches / larger batches) — decision needed.
2. Then decide the SEP probe target (discrete vs weighted) and the feature set for logistic regression (VIF).
