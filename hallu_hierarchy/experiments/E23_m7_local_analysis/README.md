# E23 — M7 local analysis of the E22 run (S124, seed 0)

**Date** 2026-11-26 · local CPU, no Kaggle · ~5 min · inputs: kernel v16 outputs (results of all 9 checkpoints,
last checkpoint weights, its hidden states, people table, exposure log; in `outputs/m7_v16/`, gitignored) ·
code `m7_synthetic/analysis/local_analysis.py` · full text output `outputs/m7_v16/analysis/ANALYSIS.md` ·
figures `figs/`.

## Why
E22 left two odd results (19% correct after one exposure; completion far below QA) and the questions about
detection: does entropy separate FA / AH / UH (H13), and is anything stored but not extracted (H19)?

## Results
**1. Completion vs QA: a scoring artefact.** After "{name} was born in" the model's top token is " the" (as in "the
city of …") for 3,155 of 4,600 people. Scored as "most likely city", completion = QA:

| exp_birth | 1 | 3 | 10 | 30+ |
|---|---|---|---|---|
| QA | 19 | 67 | 99 | 100 |
| completion, most likely city | 15 | 61 | 99 | 100 |

Name tokenization (with or without a leading space or `<eos>`) changes nothing. Storage and extraction agree at
the last checkpoint, so there is **no stored-but-not-extracted gap** here (H19, as in Allen-Zhu's results).

**2. Recency explains exp 1.** The test is QA accuracy as a function of when the last birth-city sentence
appeared in the single training pass. AUROC is 0.73 at exp 1 and 0.67 at exp 3.

| position of last exposure | 0–20% | 20–40% | 40–60% | 60–80% | 80–90% | 90–100% |
|---|---|---|---|---|---|---|
| exp 1 FA % | 1 | 3 | 23 | **41** | 26 | 24 |
| exp 3 FA % | (n=4) | 6 | 45 | **78** | 77 | 74 |

Facts seen early are forgotten, and facts seen in the last fifth are not yet fully learned (lr has decayed).

**3. Learning curves.** Exp 300 reaches 94% by 50M tokens and exp 10 reaches 99% only at 249M. Exp 3 is still
rising at the end (67%). So learning is late *for rare facts*, not for frequent ones.

**4. Entropy at exp 1–3** (the only exposures with errors; n = 1,520: FA 690, AH 128, UH 702). AUROC for the QA
prompt, pooled [95% CI], with the values within exp 1 / exp 3:

| | token entropy | city entropy | semantic entropy (10 samples) | −margin |
|---|---|---|---|---|
| FA vs wrong | 0.84 [0.81, 0.86] (0.74 / 0.87) | 0.84 | 0.80 | 0.84 |
| FA vs AH | 0.77 [0.73, 0.81] (0.67 / 0.78) | 0.77 | 0.74 | 0.79 |
| AH vs UH | 0.63 [0.57, 0.68] | 0.63 | 0.60 | 0.60 |

**5. Probes fail (checked, not a bug).**
- Linear correctness probe on the last QA-prompt token: AUROC 0.46–0.52 at every layer. The same saved states
  reproduce entropy AUROC 0.84 and 93% of the answers through the unembedding.
- Q-probe (birth city from the name-only state, 200 classes): 0.7–1.3%, against 0.5% chance.
- The logit lens on the name state ranks the birth city at about 84 of 200.

**6. AH by familiarity (H12).** AH share of wrong answers at ratio 30 is **37% for high familiarity vs 19% for
low**; at ratio 1 it is 15% vs 5%. By exp: ratio 30 gives 25% at exp 1 and 34% at exp 3.

## Insights
- **H11 confirmed and H12 supported** (one seed): AH needs a strong competitor *and* a familiar subject.
- **H13 / H2 do not hold in this setting.** AH answers here are *not* confident: entropy separates FA from AH at
  0.77, which is better than on Qwen × PopQA (0.65). In a 93M model trained on one pass, AH is "not sure, so
  fall back to the most associated city". It is not the confident, consistent error that H2 assumes. To create
  confident AH, the competitor must be learned *as the answer* (e.g. reverse links or conflicting statements).
  This is a design question for the next run.
- The failed probes are a finding: confidence is readable from the softmax (non-linear) but not by a linear
  probe at this model size and n ≈ 1,300. Storage sits on the relation path, not on the name token alone, so
  Allen-Zhu's Q-probe needs their fine-tuning variant, not a plain linear probe.
- The exposure counter is not the whole story: *when* a fact was seen matters as much as how often (recency).
  Future analyses should control for the position of the last exposure.
- Threats: one seed; errors only at exp 1–3 (n AH = 128); probe layer chosen on test folds (no effect, it is at chance).

## Next steps
1. Slides: `figs/accuracy_vs_exposure.png`, `figs/ah_vs_ratio.png`, `figs/learning_curves.png`, `figs/recency.png`
   plus the entropy table.
2. Download 2–3 earlier checkpoints and repeat the completion vs QA check (H19 over training, no Kaggle run).
3. Next Kaggle run (needs the user's yes): a finer grid exp {1, 2, 3, 5, 7, 10}, conflicting competitor statements
   (for confident AH), 2 more seeds. Also run this analysis inside the `m7` stage on Kaggle.
