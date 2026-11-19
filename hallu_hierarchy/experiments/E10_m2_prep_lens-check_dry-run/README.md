# E10 — M2 prep: final-norm check + local dry run (no Kaggle run)

2026-11-19 · local CPU · inputs: `features.npz`, `generations.jsonl`, `questions.jsonl` downloaded from
`hallu-hierarchy-m-1-5` · code: uncommitted M2 draft on top of 501319f

## Why
Before trusting layer-wise features: (1) is hidden-state index 0 the embedding layer, and does index 36 already
have the final layer norm (the logit lens skips the norm there)? (2) Does M2 run end to end on the real
PopQA outputs, with the right labels, before spending GPU time?

## Setup
(1) Compare the logit lens at indices 36 and 35 (as saved by generation) with the real first-token
distribution of the greedy pass, for all 14,267 questions. Source check of transformers 4.x (Kaggle) and 5.x.
(2) `correctness,entropy` on a 200-question subset with a stand-in NLI (exact match instead of DeBERTa).

## Results
| Lens read at | top-1 = real first token | median \|ΔH\| vs real entropy |
|---|---|---|
| **index 36, no norm** | **99.77%** | **0.0000** (max 0.08) |
| index 35, final norm applied | 31% | 1.25 |

37 entries per question = embeddings + 36 blocks → index 0 is the embedding output.
Dry run: 4 s on CPU; labels regenerated in the run (= E09), all entropy checks pass; 36 eligible correct of 200.
Stand-in AUROCs (wrong vs correct): uncertainty levels 0.85–0.87; attention to subject 0.17–0.27 (higher for
correct answers); `lens_H_18/24/30` ≈ 0.4–0.58. Token, sequence, semantic and lens_H_36 entropies correlate
0.85–0.90 (VIF 6–8).

## Insights
- Index 36 is already final-normed (both transformers 4.x and 5.x put `last_hidden_state` there); the existing
  logit lens is correct. 0.23% top-1 mismatches are fp16 near-ties.
- Without running `correctness` in the same run, M2 would have read the old pre-M1.6 labels — now guarded.
- The final-layer uncertainty measures are nearly redundant (VIF 6–8): rule 4's correlation/VIF report matters.
- Mid-layer lens entropy carries little signal on its own; l* and attention look more informative (stand-in only).

## Next steps
1. M2 on the 200-question subset on Kaggle with the real DeBERTa-MNLI (`--stages correctness,entropy --gpu on`).
2. If the semantic clusters look sensible, M2 on all questions (NLI ~25–40 min).
