# E18 — Diagnostics: three-class models and where the correctness probe's FA vs AH signal comes from

2026-11-23 · code at commit ab121af · local CPU, analysis scripts in this folder (`m5_diag.py`, `m5_flat.py`, run from
the repo root with `PYTHONPATH=.`) · inputs: M4 per-layer OOF scores (`probes/oof_by_source.npz`, dataset
`correctness-probes`), M3 labels, M2 features · no report file (printed output, numbers below)

## Why
After E17: why does the correctness probe separate FA vs AH (0.80) but not AH vs UH (0.65)? Does a 3-class model
with the probe do better? (Discussion with the user; serves H2/H3.)

## Results (within-relation AUROC)
| check | FA vs AH | AH vs UH |
|---|---|---|
| correctness probe, last prompt token (best layer 36) | 0.698 | 0.628 |
| correctness probe, answer token (layers 20–28) | 0.80 | 0.64–0.66 |
| SEP, answer token (layers 20–28) | 0.66–0.67 | **0.75** |
| "default answer" frequency score | 0.584 | 0.438 |
| correctness probe with the identical answer string (4,089 pairs) | 0.61 / 0.60 (SLT / TBG) | |

3-class one-vs-rest, macro within: SEP only 0.741 · correctness probe only 0.792 · **SEP + correctness probe
0.814** · E+L+A+F 0.788 (GB) · E+L+A+F+C 0.828 (GB). AH one-vs-rest within: 0.57 → 0.68 → 0.71.
Median scores: SEP FA 0.03 / AH 0.15 / UH 0.95; correctness probe FA 0.08 / AH 0.64 / UH 0.97.

## Insights
- **Two axes:** uncertainty (SEP) splits UH from {FA, AH}; the correctness probe splits FA from {AH, UH}. AH = confident
  but wrong. Two scores reach 0.814 of the full model's 0.828 in 3 classes.
- The FA vs AH signal appears mostly **after the answer token** (0.70 → 0.80), and a good part of it is
  **answer identity** (same answer string: 0.61). It is not "default answers". A residual subject–answer
  consistency signal of about 0.60 remains.
- The correctness probe can't split AH from UH because both were "wrong" in its training target.

## Next steps
See `../HYPOTHESES.md` section 3: an AH-vs-FA hidden-state probe (baseline), a relation–answer type-match feature,
answer–subject association strength (bridge to H9), attention at the answer position, and a closed → open transfer
test of the correctness probe. Then M6.
