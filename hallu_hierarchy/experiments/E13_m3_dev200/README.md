# E13 — M3 dev run: AH/UH via Wikipedia co-occurrence on 200 wrong + 50 correct answers

2026-11-21 · commit 61bce02 · stage labeling · **local CPU, no GPU** · 78 min (77 min of it rate-limited search) ·
**STATUS ⚠️ WARN** (only: AH = 11 < 30, expected on 200 rows) · Report: [`reports/2026-11-21_qwen2.5-3b_popqa_full_labeling_local-E13.md`](../../reports/2026-11-21_qwen2.5-3b_popqa_full_labeling_local-E13.md)

## Why
First M3 run (§8 step 6). Does the Wikipedia proxy work at all, how do the cutoffs compare, and what does
the ambiguity check do, before running on all ~11k wrong answers (H2/H3 need AH labels).

## Setup
New `labeling` stage. Seeded subset of 200 eligible wrong + 50 eligible correct answers. Labels regenerated
locally (identical to kernel v8). Signals: `cooc_page` (answer on the subject's page), `cooc_count` (pages that
mention the answer and link to the subject page, `linksto:` search, + the page itself), gold-answer versions of
both as a check. Ambiguity: disambiguation page for the base name exists / answer appears on it.

## Results
| | page rule | count≥1 | count≥3 | count≥10 |
|---|---|---|---|---|
| AH share of wrong (before ambiguity check) | **8.5%** (17) | 42% (85) | 25% (50) | 15% (29) |
| agreement with page rule | – | 66% | 84% | 92% |
| AH after `disambig_exists` / `answer_on_disambig` | 11 / 16 | 42 / 80 | 25 / 46 | 14 / 28 |

- **Proxy check passes:** the gold answer is on the subject's page for **96.5%** of wrong rows (100% of correct
  rows), so "the model's answer is not on the page" is real evidence. Correct answers: 94% on page.
- cooc_count median: wrong answer 0, gold answer on the same rows 17, correct answers 73.
- AH by group (page rule): **closed 17 of 51 wrong (33%), open 3 of 149 (2%)**. AH is mostly genre, religion, capital,
  place of birth. Open-set relations (director, author, producer, screenwriter): 0%.
- Disambiguation page exists for **61%** of wrong subjects; the model's answer is on it for only 3%.
- Search is rate-limited for anonymous clients: **~10 s per search** (HTTP 429, Retry-After 30). Page fetches are batched (50 titles per request).

## Insights
- **The page rule is precise.** All 17 page-AH look like genuine associated hallucinations (Prague for a Czech
  person's birthplace, Celle as the capital of Lower Saxony, George Martin for *Let It Be*). Page-AH is a strict
  subset of count≥3-AH (0 page-AH missed by count≥3).
- **Count rules add recall but also noise.** Count≥1 or ≥3 adds famous names reached through list or award pages
  ("Guillermo del Toro" for *Hugo*, "David Fincher" for *Stay*), next to plausible ones ("Cronus" for the Horae).
  The manual check has to decide. Counts also cannot scale: 11k searches × 10 s ≈ 30 h.
- **`disambig_exists` is too blunt:** it removes 35–50% of AH, including unambiguous cases (Lower Saxony,
  Muhammad Ali, *Let It Be*). `answer_on_disambig` removes only answers that fit another same-name entity
  (*Caught*, "Drama"). This is the §9 decision; both are reported.
- **Threat to H2/H3: on PopQA × Qwen-3B, AH is rare (~8% of wrong ≈ 900 rows) and almost all closed-set.** AH vs
  FA would then mostly compare relation types (closed-set questions are also easier, H5). Within-relation AH vs FA is
  possible for only a few relations (genre). TriviaQA / EntityQuestions (more popular subjects, AH expected more
  often) matter more than planned.

## Next steps
1. **E14: full run with the page rule only** (`linked_counts: false`: pages + disambiguation pages, no search;
   ~600 batched requests). Count cutoffs stay compared on these 200 rows. Gives the AH count per relation.
2. **User decision (§9): ambiguity rule.** Recommendation: `answer_on_disambig` as the rule, `disambig_exists` as
   a sensitivity analysis. Switching it later is an offline rerun (cached).
3. **E15: manual check of ~100** rows (stratified AH/UH × open/closed from E14) → page rule precision and recall.
4. Start TriviaQA generation + EntityQuestions pilot on Kaggle (GPU) soon: PopQA alone may not give enough AH
   for H2/H3 outside closed-set relations.
