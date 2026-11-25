# E20 — M7 data generator, token budget, 1% dry run

**Date** 2026-11-24 · **commit** this commit (parent e44b092) · local CPU, no Kaggle · 21 s ·
generator `gen-v0.1`, templates `templates-v0.1`, seed 7020 · outputs (gitignored):
`outputs/m7_synthetic/E20_dryrun_1pct/` · plan: `../../M7_SYNTHETIC_EXPOSURE_PLAN.md` §1, decisions D1–D9.

## Why
First step of M7 (plan §8): build the synthetic biography generator with controlled exposure counts, print the
token budget of the full design, and check the output on a 1% corpus before any training run.

## Setup
- `m7_synthetic/data/generate.py` + `config_v0.yaml` + frozen `templates.json` (35 + 5 held-out statements per
  attribute; 6 + 4 questions) + `lexicon.py` (invented people, 200 cities, 60 universities, 120 employers).
- Full design: 7 exp_birth × 4 ratio × 2 familiarity × 2 splits × 150 people = 16,800 people.
- Dry run: `--fraction 0.01` → 2 people per cell × split = 224 people (every cell present, full per-person counts).
- Command: `python -m hallu_hierarchy.m7_synthetic.data.generate --fraction 0.01 --out hallu_hierarchy/outputs/m7_synthetic/E20_dryrun_1pct`

## Results
| | dry run (224 people) | full design (calibrated) |
|---|---|---|
| documents | 376,990 | — |
| tokens | 9.47 M | **≈ 681 M** (estimate before writing: 708 M) |
| fact exposures | 245,984 | 18.4 M sentences → 298 M tokens |
| balancing name mentions (D2) | 189,757 | 13.2 M → **380 M tokens (56%)** |
| QA pairs | 1,932 | 145 k → 2.4 M tokens |
| tokens per exposure / balancing doc / QA | 16.2 / 28.8 / 16.9 | |

Full design by exp_birth: 1: 10M · 3: 14M · 10: 26M · 30: 62M · 100: 188M · 300: 179M · 1000: 201M.
By ratio: 0: 206M · 1: 195M · 10: 150M · 30: 129M (low ratios carry more balancing text).

Checks on the dry run (all pass; also as unit tests in `tests/test_m7_generator.py`):
- logged exposure counts = planned counts for every (person, attribute);
- birth, work, university and employer city all differ; QA only for qa_train; no work-city QA at ratio 0;
- name mentions equal across ratios within (exp_birth, familiarity), e.g. 4,400 at exp 1000 / high;
- chance-level AH (linked cities ÷ 199): mean 3.8%, max 4.5% (7.6 linked cities per person);
- exposures spread uniformly (exp-1000 birth sentences at 10/50/90% of the corpus: 0.10 / 0.50 / 0.90);
- tokens on disk = sum of document lengths, every document ends with `<|endoftext|>`; output is deterministic.
- 1,800 of 16,800 people are in capped cells (D3).

## Insights
- **The full design is ≈ 2.3× the 200–300M target.** Facts alone (298 M) fit the target; name balancing (D2)
  more than doubles it, mainly through the generic filler sentences around each name mention.
- The cost sits in exp_birth 100–1000 (84%), as the plan expected.
- BPE trained on one biography per person + templates: vocab reached only 5,143 of 8,192 (corpus too small for more
  merges); a full name is 3 tokens (one per invented word), a city 1 token. Acceptable; record for E21.

## Next steps
1. **Decide the cut (user):** e.g. balancing mentions without generic filler sentences (≈ 12 instead of 29 tokens
   → ≈ 460 M total), and/or 75 instead of 150 people per split in the exp 300 / 1000 cells, and/or drop 1000.
2. E21 S25 pilot on a small corpus (Kaggle; needs the user's explicit yes): tokens/s and sanity checks §3.
3. Before E21: verify the Allen-Zhu Part 3.1 QA mixing recipe (D6).

## E20b — budget cut and city-frequency balancing (2026-11-25, local, generator `gen-v0.2`, `config_v1.yaml`)
Decisions D10–D14 in the plan. Changes: exp 1000 dropped; balancing mentions without filler; cities assigned so
every city is mentioned about equally (birth, work, university, employer; org sentences weighted by the 40% of
their templates that name the city).

| | gen-v0.1 | gen-v0.2 |
|---|---|---|
| people | 16,800 | 14,400 |
| tokens, full design (calibrated) | 681 M | **335 M** |
| city mentions max / median (full design) | 1.52 | **1.01** |
| chance-level AH, 10% corpus: uniform / frequency-weighted | — | 0.038 / 0.038 |
| 10% corpus (pilot size) | — | 34.5 M tokens, 35 s |

- Balancing only works with enough people: a 1% subset stays unbalanced (max/median 17), 10% is balanced → pilot uses 10%.
- With balanced cities, frequency-weighted chance AH = uniform chance AH, so a "guess a frequent city" model can no
  longer produce AH above chance.
- Next: E21 S25 pilot on the 10% corpus (Kaggle, needs your yes) after the training code exists.
