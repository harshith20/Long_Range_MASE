# E17 — M5 classifier, ablations and baselines (PopQA × Qwen2.5-3B)

2026-11-22 · code after commit bd97535 (uncommitted at run time) · stage classifier · **local CPU** · 3.4 min ·
**STATUS ⚠️ WARN** (the warnings are results, see below) ·
Report: [`reports/2026-11-22_qwen2.5-3b_popqa_full_classifier_local-E17.md`](../../reports/2026-11-22_qwen2.5-3b_popqa_full_classifier_local-E17.md)
(header says "M4": the config milestone was still M4)

## Why
M5 (§8 step 8): does the learned boundary on single-pass features detect hallucinations (H1), beat sampling
(H1b), fail on AH (H2), get recovered by non-confidence signals (H3)? H5 via within-relation, LORO and relation-only.

## Setup
Rows: final M3 labels (E14b; **no manual check**, user decision) → FA 2,128 · AH 740 · UH 10,094. Inputs are the
M4 out-of-fold probe scores (dataset `correctness-probes`) + M2 single-pass features. Feature sets E (token entropy
+ SEP probes + ratio) → +L (logit lens) → +A (subject attention) → +F (familiarity) → +C (correctness probe).
LR (C on an inner subject split) and HistGradientBoosting (fixed settings), balanced weights, same 5 subject folds.

## Results (within-relation macro AUROC; pooled in brackets)
| task | E | E+L+A+F (LR / GB) | + correctness probe | sampled SE | SEP alone | correctness probe | relation-only |
|---|---|---|---|---|---|---|---|
| FA vs AH+UH (13 rel.) | 0.833 | 0.843 / **0.860** (0.941) | 0.905 | 0.787 | 0.833 | **0.900** | 0.447 (0.852) |
| AH vs UH, pos UH (7) | 0.730 | **0.738** / 0.728 (0.895) | 0.745 | 0.683 | 0.732 | 0.653 | 0.434 (0.836) |
| FA vs AH (7) | 0.684 | 0.674 / 0.692 (0.798) | **0.807** | 0.648 | 0.646 | 0.800 | 0.428 (**0.754**) |

3-class: flat LR macro AUROC 0.891 (AH 0.815 pooled / 0.629 within) beats the two-stage composition at 0.846
(AH 0.671 / 0.518). LORO (main LR): hall 0.840 (religion 0.56, occupation 0.76); FA vs AH 0.648.
Without rows that carry the heuristic ambiguous_subject flag: hall within 0.802, FA vs AH 0.619.
Cost after the greedy pass: probes + classifier ≈ 0.015 ms/question (CPU), vs NLI alone 0.375 s/question plus 10 samples.

## Insights
- **H1 / H1b hold:** the single-pass hierarchy (GB, 0.860 within) beats sampled semantic entropy (0.787) by
  0.07, and it runs about 4 orders of magnitude faster after the greedy pass.
- **Negative result (report it):** the hierarchy does **not** beat the direct correctness probe (0.900). Adding
  the probe as a feature gives 0.905, so the entropy features add almost nothing on top of it.
- **H2 is supported:** entropy-only FA vs AH is 0.68 within-relation (vs 0.83 for FA vs hallucination), and only
  0.60–0.63 in open-set relations.
- **H3 is negative for the planned signals:** logit lens, attention and familiarity do not raise FA vs AH within
  relation (0.684 → 0.674 / 0.692). The hidden-state correctness probe does (0.80). AH is visible in the hidden
  state but not in the hand-made non-confidence features.
- **H5 is strong for AH:** for FA vs AH the pooled AUROC of the main model (0.757 LR) equals relation-only
  (0.754). Every pooled number for AH is about question type. Only within-relation and LORO numbers count.
- **H4 shows up in the classifier:** familiarity is the feature that helps AH vs UH (largest negative LR
  weight: familiar subject → AH), and it adds nothing to FA vs AH.
- Two-stage is worse than flat for AH: errors of stage 1 multiply into P(AH). Report flat as the main 3-class model.
- ECE 0.09–0.25: the balanced class weights shift the probabilities. Calibrate (e.g. Platt on the inner split) if
  probabilities are reported.
- Threats: M3 has no human check; 7 relations / 740 AH; one model and one dataset (H8 untested).

## Next steps
1. **Thesis framing:** the main claim is H1b (single pass ≥ sampling, at negligible cost) and H2 (entropy can't
   see AH). The correctness probe is the honest upper reference. H3 is "not with these features".
2. **TriviaQA generation + EntityQuestions pilot** (Kaggle GPU; needs an explicit go for `generation`). This tests
   H8 and gives more open-set AH, the weakest cell.
3. Dedicated timing run on the T4 (greedy only vs greedy + 10 samples + NLI) for the final H1b cost table.
4. Optional: an AH-vs-FA hidden-state probe as an H3 upper bound (a baseline only); calibration of the main
   model; M6 OLMo + infini-gram (H9, proxy validation).
