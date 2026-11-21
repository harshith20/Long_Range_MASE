# E14 — M3 on all PopQA wrong answers (page rule + ambiguity check)

2026-11-21 · commit 106722b · stage labeling · **local CPU, no GPU** · 44 min first pass (network), reruns ~5 s
from the cache (78 MB) · **STATUS ✅ PASS** · Report: [`reports/2026-11-21_qwen2.5-3b_popqa_full_labeling_local-E14.md`](../../reports/2026-11-21_qwen2.5-3b_popqa_full_labeling_local-E14.md)

## Why
AH/UH labels for all 11,113 eligible wrong answers (§8 step 6). This gives the class sizes for H2/H3, shows which
relations allow a within-relation AH vs FA test (H5), and provides the rows for the manual check (E15).

## Setup (vs E13)
All rows, page rule only (`linked_counts: false`, because search is rate-limited, see E13). Sentence detection fixed
after the first pass: a row counts as a sentence only if it contains sentence words ("is", "was", "has", "not", …).
Before the fix, long names like "Thomas Howard, 3rd Duke of Norfolk" (father) were wrongly left unscored.

## Results
| | count |
|---|---|
| wrong scored / unscored (no page 33, sentence 39) | 11,041 / 72 (0.6%) |
| gold answer on the subject page (wrong rows / correct rows) | **97%** / 100% |
| AH before the ambiguity check (page rule) | **947 (8.6% of scored wrong)** |
| AH after `disambig_exists` (current default) / after `answer_on_disambig` | 510 / **740** |
| FA / UH | 2,128 / 10,094 |

AH (before ambiguity check) by relation: genre 331, place of birth 106, religion 77, capital of 69, capital 67,
occupation 60, father 55; the person-name relations about works (director, screenwriter, producer, author,
composer) only 19–29 each (1–3% of their wrong answers), sport 8.

**Within-relation AH vs FA possible (≥ 30 each) for 7 relations under either ambiguity rule:** genre, capital,
capital of, religion (closed) and place of birth, occupation, father (open).

## Insights
- **Proxy confirmed at scale:** gold on page 97% overall, 95–100% in every relation except religion (86%), so UH is reliable.
- **AH is rare on PopQA × Qwen-3B (8.6% of wrong) and depends on the relation:** 26–62% for capital-type relations,
  genre and religion, 13–21% for family / birthplace / occupation, 1–3% for creators of works. For long-tail films
  and books the model guesses a famous name that has nothing to do with the work, which is UH by definition.
- **The relation confound is smaller than E13 suggested:** AH is 39% open-set vs 24% for FA, and 7 relations
  support a within-relation test. The H2/H3 analysis must still report within-relation results.
- **`disambig_exists` hits works hardest** (64–77% of film/book subjects have a disambiguation page vs 13–27%
  for people and places). Under this rule, AH for creators of works falls to 4–17 per relation. The model's
  answer actually appears on the disambiguation page for only 5% of rows.
- 39 sentence answers left unscored, mostly missed refusals ("X has no known father"). A few contain real
  answers ("Ronan Farrow's father is Woody Allen"); too few to matter.

## Next steps
1. **User decision (§9): ambiguity rule.** Recommendation: `answer_on_disambig` (740 AH), with `disambig_exists`
   (510 AH) as a sensitivity analysis. Offline rerun after the change.
2. **E15: manual check.** `overrides/m3_manual_popqa_qwen2.5-3b.csv` holds 100 rows, ~16 per cell
   (AH / UH / ambiguous × open / closed). The user fills `manual_label` (AH / UH / not_wrong / ambiguous /
   unclear), then an offline rerun reports agreement and AH precision/recall of the page rule.
3. Optional: add the clearest sentence patterns ("has no known", "was not") to `refusal_patterns.yaml` in the
   next correctness rerun.
4. Then M4 (probes); TriviaQA / EntityQuestions on Kaggle for more AH in open-set relations.
