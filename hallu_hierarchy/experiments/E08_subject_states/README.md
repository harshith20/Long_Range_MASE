# E08 — Subject-state stage (H4) on all PopQA

2026-11-18 · commit 2100edc · stage subject_states · GPU on (T4) · kernel v7 · ~15 min · **STATUS ✅ PASS**
Report: [`reports/2026-11-18_qwen2.5-3b_popqa_full_subject_states_v7.md`](../../reports/2026-11-18_qwen2.5-3b_popqa_full_subject_states_v7.md) · follows [E07](../E07_m1-6c_folds_refusals/)

## Why
H4 needs the hidden state at the subject's last token (prompt-only pass) to train the subject-familiarity probe.

## Setup
New stage `generation/subject_states.py`: one forward pass over the prompt only, same prompts and tokenizer as
generation, states cached at the same layers, checkpoints every 100 questions.

## Results
| | |
|---|---|
| questions done | 14,267 / 14,267 (143/143 checkpoints) |
| time | 0.057 s/question (~15 min; generation was 0.74 s) |
| subject found | 14,264 (3 missing) |
| prompts / subject positions vs generation | identical for all questions |
| last-prompt-token state vs generation (20 questions) | max relative diff 0.0019 |
| states | `[14267, 10, 2048]` fp16 at hidden-state indices 0, 4, …, 32, 36 |

## Insights
- The stage reproduces generation's prompts exactly, so subject states line up with the generation features.
- Index 0 is the embedding output (HF `hidden_states[0]`), not a transformer block.
- Open question for M2: whether index 36 already has the final layer norm. `generate.py` `logit_lens` assumes it does.
- Storage: saving this output as a new version of `hallu-hierarchy-m-1-5` *replaced* the generations
  (dataset v2 = subject states only). Fixed after the fact: v3 = generations again; subject states go to their
  own dataset `hallu-hierarchy-subject-states`.

## Next steps
1. Regenerate the final labels with folds from the generations (E09) — they were in no dataset.
2. M2 on a 200-question subset; check the index-36 layer-norm question first.
