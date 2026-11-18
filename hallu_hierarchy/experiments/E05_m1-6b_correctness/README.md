# E05 — M1.6b: religion head words + relation / popularity checks

2026-11-18 · commit 06bf86f · stage correctness · GPU off · kernel v5 · ~3 min · **STATUS ⚠️ WARN** (expected)
Report: [`reports/2026-11-18_qwen2.5-3b_popqa_full_correctness_v5.md`](../../reports/2026-11-18_qwen2.5-3b_popqa_full_correctness_v5.md) · reruns [E04](../E04_m1-6_correctness/)

## Why
Fix the religion partials found in E04, and add three report checks before M2: refusals per relation,
test-split class counts per relation (can H5 be tested within relation?), accuracy by subject popularity (H4).

## Setup
vs E04: `church`, `religion` added as head words; report-only additions (no other label rule changed).

## Results
| | E04 | E05 |
|---|---|---|
| correct | 2,047 (14.3%) | **2,128 (14.9%)** |
| partial | 165 (1.2%) | **86 (0.6%)** |
| abstain | 787 | 787 |
| religion accuracy | 13.3% | 37.3% |

| Within-relation usable (≥ 30 correct and ≥ 30 wrong in test) | |
|---|---|
| Closed-set | 5 of 7: genre, country, capital, sport, religion |
| Open-set | **0 of 9** (test correct: author 22, occupation 18, place of birth 16, director 8, producer 4, screenwriter 3) |
| Group level, test correct/wrong | closed 340 / 470 · open 92 / 1,783 |

Accuracy by `s_pop` decile (low → high popularity): 20 · 11 · 9 · 8 · 7 · 8 · 9 · 11 · 17 · 47 %
(closed-set 40% in the lowest decile vs 21–31% in deciles 2–8; open-set 2–5% until decile 9, 21% at the top).

## Insights
- The religion fix worked as intended (+81 correct, partials halved); remaining partials are genuinely less
  specific ("Christian" vs "christian faith", "Ukraine" vs "volyn ukraine").
- **The WARN is model behaviour, not a rule bug:** `capital of` refusals are "Not applicable" (67) and premise
  denials like "Lund is not the capital of any country" (49). `producer` refusals are almost all
  "Not enough information". No real answers were seen among the audit examples.
  Open question: a premise denial contradicts the gold answer — abstain (current) or wrong?
- **H5 cannot be tested within open-set relations with a single 20% test split.** All 5 usable relations are
  closed-set; within-relation results would describe only the "easy, guessable" relations.
- **Popularity is U-shaped, not monotonic.** The least popular decile is *more* accurate than the middle,
  driven by closed-set questions (40%). Likely guessing of frequent answers (e.g. a common country or sport)
  for obscure subjects → some FA labels may be lucky guesses rather than factual recall. This matters for H4
  (familiarity ≠ correctness at the low end) and for what "FA" means.
- Real recall effect only at the top: decile 10 reaches 75% closed / 21% open.

## Next steps
1. **Evaluation design (decision needed before M5):** subject-grouped 5-fold cross-validation instead of one
   test split. Every question is then tested once: open-set relations with ≥ 30 correct in total (author 107,
   occupation 90, place of birth 71, father 61, composer 52, director 49, screenwriter 31) become usable.
2. **Guessability baseline (CPU, cheap):** per closed-set relation, accuracy of always answering the most common
   gold answer, overall and in the lowest `s_pop` decile — to see how many FA labels a guesser would get.
3. Decide premise denials in `capital of` (keep `abstain`, or a separate tag) — small, can wait until M3.
4. Then subject-state stage (H4, GPU) → M2 on a 200-question subset.
