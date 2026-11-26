# E21 — M7 Kaggle smoke test (S124 on 2 × T4, 1% corpus)

**Date** 2026-11-26 · kernel `hallu-hierarchy-m1-5` v13 (FAIL) and v14 (WARN) · commits 47840ce, 8db4caf ·
stage `m7`, config `configs/m7_smoke.yaml` · GPU on (T4 × 2) · v14: 7.4 min · report
`reports/2026-11-26_m7_smoke_m7_v14.md` · generator gen-v1.0, seed 7020, train seed 0.

## Why
Check that generation → training (DDP on 2 GPUs) → checkpoints → evaluation runs on Kaggle, and measure tokens/s
to plan the full S124 run (plan §8; D15–D18). Not a results run: 150 steps cannot teach facts.

## Results
| | v13 | v14 |
|---|---|---|
| status | ❌ crashed at launch in 13 s (torchrun got the module name without `-m`) | ⚠️ WARN (accuracy check only) |
| model | | 92.8M params (85.1M non-embedding), 2 GPUs, steps 150 × 16,384 tokens |
| corpus | 2.5M tokens, 92 people, 10 s | same |
| training | | 408 s, final loss 0.92, median **6,590 tokens/s** |
| QA accuracy (last checkpoint) | | 2.2% (1 of 46 people; chance level) |

## Insights
- The whole path works on Kaggle with 2 GPUs: corpus, DDP training, 9 checkpoints, evaluation, report.
- **Precision bug:** training ran in bf16 because `torch.cuda.is_bf16_supported()` is true on T4, but T4
  (compute capability 7.5) only emulates bf16 → slow. Fixed: bf16 only with capability ≥ 8, else fp16.
- **Speed not measured cleanly:** the tokens/s windows contained 9 checkpoint saves (each with ~1 GB optimizer
  state) and quick evaluations within 150 steps. Fixed: checkpoint time excluded; smoke config uses 2 checkpoints.
- At 6.6k tokens/s the 270M-token full run would need ~11 h (over the 12 h session with evaluation); with fp16 a
  speed of ~25k tokens/s (≈ 3 h) is expected but must be measured.

## Next steps
1. Rerun the smoke test with fp16 and clean speed logging (needs the user's yes) → tokens/s.
2. If ≥ ~15k tokens/s: full S124 run (`configs/m7_s124.yaml`, ~270M tokens) in one session; otherwise cut
   the corpus (people per cell) or split the run across sessions (resume from `state.pt`).
