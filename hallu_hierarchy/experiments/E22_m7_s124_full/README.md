# E22 — M7 full run: S124 from scratch on the synthetic corpus (seed 0)

**Date** 2026-11-26 · commit bf5632d · stage `m7`, config `configs/m7_s124.yaml` · GPU on (T4 × 2) ·
kernel `hallu-hierarchy-m1-5` v16 · 2.0 h (corpus 11.7 min, training 1.75 h, evaluation 2.6 min) · ✅ PASS ·
report `reports/2026-11-26_m7_s124_seed0_m7_v16.md` · generator gen-v1.0 (data seed 7020), train seed 0.

## Why
This is the first real M7 result. It asks how FA/AH/UH change with exposure (exp_birth), competitor ratio and
familiarity in a model whose training data we control (H10–H15). It also asks whether QA extraction works at
all within one pass (the Allen-Zhu Fig. 1 risk). Evaluation uses 4,600 qa_test people, whose QA pairs never
appear in the training data.

## Setup
46 cells × 2 splits × 100 people = 9,200 people, 248.7M tokens, one pass, 15,181 steps × 16,384 tokens.
GPT-NeoX with rotary embeddings, 92.8M parameters, fp16 DDP. Median speed 40.1k tokens/s. Final loss 0.82.

## Results (last checkpoint, first-token answers)
| exp_birth | 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|
| QA FA % (train question template) | 19 | 67 | 99 | 100 | 100 | 100 |
| QA FA % (held-out paraphrase) | ≈ 18 | ≈ 61 | ≈ 97 | ≈ 99 | ≈ 100 | 100 |
| completion "{name} was born in" FA % | ≈ 0 | ≈ 2 | ≈ 13 | ≈ 33 | ≈ 66 | ≈ 89 |
| QA label mix AH / UH / other % | 10 / 63 / 8 | 6 / 25 / 2 | 0.1 / 0.4 / 0.1 | 0 | 0 | 0 |

AH share of wrong QA answers, by competitor ratio (chance ≈ 3.8%):

| ratio | 0 | 1 | 10 | 30 |
|---|---|---|---|---|
| AH % of wrong | 5.3 | 9.0 | 18.7 | 27.8 |

AH sources: work city 98, partner 29, employer 2.

QA accuracy over training (quick evaluation on 500 people): 1% at 12M tokens, 36% at 50M, 62% at 124M,
79% at 249M. It is still rising at the end. Completion accuracy: 0% at 12M, 26% at 87M, then flat at
22–31%.

## Insights
- **The main risk did not happen.** QA extraction works without QA pairs for the test people, and it
  carries over to a held-out question paraphrase (only a few points lower). Knowledge augmentation plus mixed
  QA training (D6) was enough.
- **The facts switch from mostly unknown to known between exp 1 and 10.** Above 10 exposures, QA is at a ceiling of
  ≈ 100%, so no errors are left to study there. All hallucinations come from exp 1 and 3. H10 (AH peaks at middle
  exposure) cannot be tested on this grid: AH is 14% of wrong answers at exp 1 and 19% at exp 3, with nothing wrong above.
- **H11 supported (one seed): the competitor effect is clean and monotonic.** As the work-city ratio rises, wrong
  answers move from unrelated cities to the competitor: AH is 5% at ratio 0 (near chance) and 28% at ratio 30,
  which is 7× chance. Nearly all AH answers are the work city, the competitor we manipulated, so AH here is
  caused by the design and is not a side effect.
- **Familiarity has almost no effect on QA**, and completion is slightly higher for *low* familiarity. This
  differs from the "celebrity helps" prior and needs a check with a finer analysis.
- **Exp 1 gives 19% correct**, far above the ≈ 0.5% chance level for picking a city. A likely cause is
  recency: single exposures that fall late in the corpus may still be remembered. This can be tested with the
  exposure log (offset of the one birth sentence vs correctness). It must be checked before anyone reads the
  exp 1 row as a stable property of the model.
- **Completion < QA is probably a measurement artefact, not "stored but not retrieved" reversed.** After
  "was born in", many statement templates continue with words such as "the city of" rather than the city
  itself. Scoring only the first token then undercounts completion. `p_city_mass` and `rank_true` in
  results.parquet can confirm this. Until then, completion should not be used as the storage measure for
  H19. A name-only Q-probe on the saved states is the better storage test.
- Threats: one seed only; first-token scoring only; coarse grid in the important range from exp 1 to 10.

## Next steps
1. Local analysis of the downloaded results.parquet and states.npz, if the user agrees (no Kaggle):
   - recency test for exp 1 and 3;
   - `p_city_mass` check of the completion prompt;
   - entropy AUROC for FA vs hallucination and FA vs AH, restricted to exp 1–3 (the only exposures where
     errors exist);
   - Q-probe on the name state for H19 (stored vs extracted).
2. Presentation material: the accuracy-vs-exposure curve, the AH-vs-ratio bar chart and the checkpoint curve.
3. Later, if time allows (Kaggle, needs the user's yes): a finer grid exp {1, 2, 3, 5, 7, 10}, and 2 more seeds.
