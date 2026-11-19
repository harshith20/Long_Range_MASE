# E12 — M2 on all 14,267 PopQA questions (entropy hierarchy)

2026-11-19 · commit 04dc528 · stages correctness, entropy · GPU T4 · kernel v11 · ~90 min (NLI 89 min) ·
**STATUS ⚠️ WARN** (expected: `capital of` refusals only) · Report: [`reports/2026-11-19_qwen2.5-3b_popqa_full_correctness+entropy_v11.md`](../../reports/2026-11-19_qwen2.5-3b_popqa_full_correctness+entropy_v11.md)

## Why
All M2 features on the full set (§8 step 4). This is the first run with **within-relation** AUROCs (H5), and it
decides the SEP probe target (discrete vs weighted SE) and the logistic-regression feature set.

## Setup (vs E11)
All questions; NLI in fp32, batch 128, length-sorted batches (no autocast). Labels identical to E07/E09.

## Results (single-feature AUROC, wrong vs correct; within-rel = macro over 13 relations)
| Feature | Pooled | Within-relation | Gap |
|---|---|---|---|
| tok_max | **0.906** | **0.797** | 0.11 |
| tok_mean | 0.893 | 0.785 | 0.11 |
| lens_H_36 | 0.900 | 0.783 | 0.12 |
| sem_entropy (discrete) | 0.898 | 0.781 | 0.12 |
| sem_entropy_w (weighted) | 0.892 | 0.775 | 0.12 |
| seq_nll | 0.892 | 0.766 | 0.13 |
| attn_mean_32 (subject attention) | 0.172 (≈0.83 flipped) | 0.317 (≈0.68 flipped) | 0.15 |
| lens_lstar · lens_H_18/24/30 | 0.66 · 0.58/0.46/0.49 | 0.52 · 0.42/0.53/0.53 | – |

NLI: 677,766 pairs in 5,347 s = **127 pairs/s** (E11: 78 → 1.6× faster). Mean 6.56 clusters of 11 answers, 11% single-cluster.
VIF: sem_entropy 24.1, sem_entropy_w 13.4, seq_nll 9.3, lens_H_36 5.0, tok_mean 3.9; mid-layer lens ≈ 1.1–1.2.
Checks: lens = real output (99.77%), NLI sanity 3/3, no fp32 fallback needed, no NaN (3 attention NaN = subject not found).

## Insights
- **H5 confirmed at feature level:** every signal loses 0.11–0.15 AUROC within relation. Pooled numbers are
  partly "which relation is this" (closed-set relations are both easier and lower-entropy).
- **Greedy-pass token entropy is the best single feature** (tok_max 0.797 within-relation) and beats full-sampling
  semantic entropy (0.781). Good for H1b (cheap single-pass), bad news for "SE adds over token entropy" on PopQA:
  short entity answers rarely paraphrase, so NLI clusters ≈ distinct strings.
- **Discrete SE ≥ weighted SE** (0.898/0.781 vs 0.892/0.775) — small but consistent with E11 → discrete as SEP target.
- **Final-layer levels are one dimension** (ρ 0.74–0.94, VIF up to 24): LR must get one or two of them, not all.
- **Subject attention survives within relation** (≈0.68) and is nearly uncorrelated with entropy in principle →
  the most promising extra signal for H3; its pooled strength was half relation confound.
- Mid-layer lens entropy and l* carry ≈ no signal within relation (0.42–0.53; lens_H_18 even flips direction).
- Threat: all of this is correct vs wrong. AH vs UH (H2/H3) needs M3 labels.

## Next steps
1. Decide LR feature set: e.g. tok_max + sem_entropy (or SEP output) + attention (+ ratio), drop redundant levels;
   SEP target = discrete SE (user decision).
2. Update §0 (M2 done) — then TriviaQA generation + EntityQuestions pilot, and M3 (AH/UH labels).
3. Before M5: check subject attention per relation group (open vs closed) and against `s_pop` (H4 overlap).
