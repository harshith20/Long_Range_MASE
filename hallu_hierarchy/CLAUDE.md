# CLAUDE.md — hallu_hierarchy

This folder is a self-contained research project inside a larger repo. Work only inside
`hallu_hierarchy/` unless told otherwise. Do not modify code outside this folder.

## 0. Current status (Claude Code updates this at the end of every milestone)

- **M1 done:** Qwen2.5-3B-Instruct on 200 PopQA questions ran end to end on Kaggle in ~40 s.
  One question per subject (no leakage), no empty predictions, ~4 tokens per answer. Accuracy 17% (strict).
- **M1.5 in progress:** correctness fixes (§3). First review done — 3 disagreement rows decided (see §3, overrides).
  - First M1.5 run (before alias cleaning/overrides): strict 34 correct (17.0%), relaxed 37, 13 wrong → abstain,
    3 review rows. `ambiguous_subject` heuristic flagged 101/200 (50.5%) — too many; REPORT warns while > 30%.
  - Code for the §3/§5/§7 spec is done: generic-alias cleaning, `overrides/popqa_qwen2.5-3b.csv` (3 rows),
    `REVIEW.csv`, `REPORT.md` with STATUS/checks, `DASHBOARD.png`, fixed `kaggle/run.ipynb`, auto-discovery of
    saved runs in `/kaggle/input`. Ambiguous-flagged rows are excluded only when wrong (rule 9 is about AH/UH).
  - Generation now also saves per-head TBG attention (subject mass per layer × head, full rows at 3 layers)
    for M2 — takes effect from the full PopQA run.
- **Next:** rerun `--stages correctness` on the 200 → full PopQA generation (`configs/qwen3b_popqa_full.yaml`)
  → develop M2 on a 200-question subset.

## 1. What this project is

Thesis work: detect LLM hallucinations with **semantic entropy probes (SEPs)** and a
**multi-level entropy hierarchy**, using the entropy levels as a *joint feature vector* for a
**learned classifier boundary** instead of a hand-set threshold.

| Label | Meaning |
|---|---|
| **FA** — factual association | Model's answer matches the gold answer |
| **AH** — associated hallucination | Wrong, but the wrong answer is strongly linked to the subject (e.g. "Where was Einstein born?" → "Munich") |
| **UH** — unassociated hallucination | Wrong, and the answer has little or no link to the subject (→ "Lisbon") |
| **abstain** | Refusal ("Not enough information", "Unknown", …). Not a hallucination. Excluded from FA/AH/UH training, always counted in reports |

### Hypotheses
- **H1:** The entropy hierarchy separates UH from FA well.
- **H2:** Entropy alone largely fails on AH vs FA (AH is confident and consistent across samples → low entropy).
- **H3:** Adding non-confidence signals (layer-wise dynamics, attention to subject tokens) partly recovers AH vs FA.

A clearly documented negative result on H2 is acceptable and expected. Do not tune things to hide it.

## 2. Non-negotiable rules

1. **Labels never become features.** FA/AH/UH come from gold answers + co-occurrence. Semantic entropy is a feature, never a label.
2. **Split by subject entity, not randomly.** The same subject must never appear in both train and test.
3. **Inference must stay cheap (edge setting).** Sampling + NLI is allowed only to *create training targets* and for baselines. At inference time, only single-pass signals: probe outputs, greedy-pass token entropy, layer-wise features, attention features, classifier.
4. **Entropy levels are correlated, not independent.** Report the correlation matrix and VIF between them.
5. **Labels are per model.** Never share labels across models.
6. **Never commit outputs.** Hidden states and generations go to `/kaggle/working/outputs/` (or `outputs/` locally, gitignored).
7. **Never judge correctness by embedding similarity.** "Ulm" and "Munich" are close in embedding space, so it would mark associated hallucinations as correct.
8. **Relaxed matching never auto-promotes wrong → correct.** It only flags rows for review (§3).
9. **Flagged rows (`ambiguous_subject`, `gold_noise`) are excluded from AH/UH training** and counted in every report.
10. **The dataset name is never a feature** (it is unknown at deployment).

## 3. Pipeline

```
QA data → generation (greedy + N samples, cache hidden states + attention)
        ├→ correctness: correct / wrong / abstain (+ flags, + overrides)
        │     └→ wrong only: AH / UH via co-occurrence
        └→ entropy hierarchy: token, sequence, semantic, layer-wise, attention
              → probes on hidden states (predict each entropy level from 1 pass)
                    → boundary classifier (features → FA / AH / UH)
                          → evaluation vs baselines, per dataset, per relation
```

### Generation
- Short-form QA prompt, answer in a few words, using each model's chat template.
- 1 greedy answer (temperature 0) + N = 10 samples (temperature 1.0), max ~32 new tokens.
- Cache hidden states for the **greedy** pass at two positions:
  - **TBG**: last token of the prompt, before generation
  - **SLT**: second-last token of the generated answer
- Cache a subset of layers (every 4th + last), stored as fp16.
- Save per-token entropy of the greedy answer and log-probs of all samples.
- Save the attention row of the last prompt token at selected middle/late layers (all heads), plus the subject's token positions in the prompt (via tokenizer offsets).

### Correctness
Runs on saved generations; no GPU needed. Applied to the greedy answer **and** to every sample.

1. **Normalize** both strings: lowercase, strip punctuation and articles (a/an/the).
2. **Alias cleaning:** drop gold aliases that are a single generic word (e.g. "church", "film", "music", "movie", "series").
   The list lives in `configs/generic_aliases.yaml`. Log every dropped alias in the report.
   Reason: PopQA lists "Church" as an alias of "Catholic Church", so any prediction containing "church" would match.
3. **Abstain:** if the answer matches the refusal pattern list → `abstain`. Seed patterns from M1:
   "not enough information", "not applicable", "unknown", "unnamed", "none", "nonegiven",
   "is not a", "various". List lives in `configs/refusal_patterns.yaml`; log every match.
4. **`match_strict`:** a normalized gold alias appears in the prediction as a whole-word span.
5. **`match_relaxed`:** also true when the prediction is a whole-word span of an alias, with
   prediction length ≥ 2 words or ≥ 6 characters (so "Drama" does not match "legal drama").
6. **`label_auto`** = `correct` if `match_strict`, else `wrong` (unless abstain). **Relaxed never changes it.**
7. **Overrides:** manual decisions live in the repo at `overrides/<dataset>_<model>.csv`
   (columns: `qid, label, gold_noise, ambiguous_subject, note`). Final `label` = override if present, else `label_auto`.
   The user gives decisions to Claude Code in VS Code; Claude Code writes them to this file and commits.
   Nobody edits files inside Kaggle.
8. **Review rows:** rows where strict and relaxed disagree, and not yet in overrides, are printed in the report
   (up to 20) and written to `/kaggle/working/REVIEW.csv`. At scale, an NLI or LLM judge replaces manual review for these rows only.
9. **Flags:**
   - `ambiguous_subject`: heuristic first pass (single-word or generic subject titles, e.g. *Emma*, *Messiah*, *Boston*, *Avatar*, *Pygmalion*). Cleared by override when the answer matches gold anyway. Later: Wikidata check of whether the model's answer is true for another entity with the same name.
   - `gold_noise`: the gold answer itself is wrong. Set by override.
10. **Initial overrides** (`overrides/popqa_qwen2.5-3b.csv`):

    | qid | label | gold_noise | ambiguous_subject | note |
    |---|---|---|---|---|
    | popqa-1919197 | correct | 0 | 0 | Avatar → "Science fiction"; short form of gold, gold refers to the film |
    | popqa-1823075 | wrong | 1 | 0 | Atlanta → "Georgia" is right in reality; PopQA gold "Fulton County" is wrong. Excluded |
    | popqa-384830 | correct | 0 | 0 | Edmund A. Walsh → "Catholic"; short form of "Catholic Church" |

11. Report what changed versus the previous run (counts moved between labels).

### AH vs UH labeling (wrong answers only, unflagged rows only)
- Co-occurrence of subject and the model's wrong answer. High → AH, low → UH.
  - OLMo: counts in OLMo's open pretraining corpus via infini-gram (check the index matches the model's training data version).
  - Qwen: Wikipedia proxy (answer appears on the subject's page / count of pages mentioning both).
- The cutoff is a parameter. Report results for 2–3 cutoffs.
- Manual check of ~100 labels; report agreement.
- **Proxy validation:** on OLMo, label with both corpus counts and the Wikipedia proxy; report agreement.

### Entropy hierarchy (features)
- **Token:** mean and max per-token entropy of the greedy answer. Mixes lexical, syntactic and factual uncertainty.
- **Sequence:** mean length-normalized NLL across the N samples. Whole-answer uncertainty; still inflated by paraphrases.
- **Semantic:** cluster the N samples by bidirectional entailment (DeBERTa-large-MNLI), entropy over cluster frequencies. Meaning-level uncertainty only. Abstaining samples form **their own cluster**. Also record `abstain_frac` among samples (training-time only).
- **Layer-wise (logit lens):** at the **last prompt token**, apply the final layer norm and unembedding to each cached layer's hidden state: $p^{(\ell)} = \text{softmax}(W_U \cdot \text{LN}(h^{(\ell)}))$. Record entropy at middle and late layers (ignore early layers) and $\ell^*$ = first layer where the final answer's first token becomes top-1.
- **Attention to subject:** with $q$ = last prompt token and $S$ = subject token positions,
  $a^{(\ell,h)} = \sum_{j \in S} A^{(\ell,h)}_{q,j}$. Use both the **mean** and the **max** over heads, at 3 middle/late layers.
- **Combination features:**
  - Logistic regression: **no difference features** (it learns linear combinations itself). Use the ratio $H_{\text{sem}} / (H_{\text{tok}} + \epsilon)$.
  - Gradient boosting: differences of z-scored levels are allowed.

### Probes
- Logistic regression probes on cached hidden states, one per entropy level, as in the SEP setup (binarize the target entropy at a training-set split point; this threshold only creates probe targets).
- Report probe AUROC per layer and per position (TBG vs SLT).

### Classifier
- **Two-stage:** (1) correct vs hallucinated, (2) AH vs UH. Plus a flat 3-class version.
- Logistic regression first (interpretable weights), then gradient boosting.
- **Class imbalance:** `class_weight="balanced"` (or equivalent) in every classifier and probe.
- Inference feature vector uses probe outputs in place of the sampled $H_{\text{sem}}$, $H_{\text{seq}}$.

### Baselines (required)
- Semantic entropy (full sampling), SEP alone, token entropy alone
- **Direct correctness probe** (linear probe predicting correct/wrong from hidden states). If the hierarchy does not beat this, say so.
- The previous single-threshold entropy-hierarchy method from earlier thesis work

### Metrics
- AUROC and AUPRC per class (one-vs-rest) and per stage; ECE.
- **Bootstrap 95% confidence intervals** on every AUROC/AUPRC (1,000 resamples, resampling by subject).
- Per-dataset, per-relation, pooled, and cross-dataset tables.

## 4. Models and datasets (Kaggle free tier: one T4, 16 GB)

Verify the exact Hugging Face IDs before use.

| Role | Model |
|---|---|
| Main | `Qwen/Qwen2.5-3B-Instruct` (fp16, ~6 GB) |
| Open-pretraining-data check | OLMo-2 1B Instruct (`allenai/OLMo-2-0425-1B-Instruct`) |
| Semantic clustering | `microsoft/deberta-large-mnli` |

No 7B models: on a T4 they need 4-bit quantization, which alters the hidden states being probed.

| Dataset | Why |
|---|---|
| PopQA | Long-tail entities; popularity score `s_pop`; relation type `prop`. Low accuracy (~17%) is expected, not a bug |
| TriviaQA (rc.nocontext) | Mostly well-known topics; more correct answers; expect more AH |
| NQ-Open | Real search queries, a realistic mix |

**Scale:**
- **Use all of PopQA (14,267 questions).** Generation is fast (~40 s per 200). Expect ~2,400 correct answers.
- The slow stage is M2 (NLI clustering over 10 samples per question) — must checkpoint and resume.

**Pooling and reporting:**
- All rows carry a `dataset` column. Classifiers are trained per dataset **and** pooled.
- Always report per dataset; never report only pooled numbers.
- **Cross-dataset transfer:** train on PopQA, test on TriviaQA and NQ-Open (and vice versa).
- Report accuracy and label counts **per relation type**; relations dominated by ambiguity or gold noise may be dropped, and any drop is documented.

## 5. Kaggle workflow

The user runs with **Save & Run All (Commit)**, so results are read from the **Logs** and **Output** tabs, not from interactive cells.

**Fixed notebook** `kaggle/run.ipynb` in the repo, 4 cells, identical every run:
1. **Setup:** clone the repo (token from Kaggle Secrets), install requirements, HF token from Kaggle Secrets.
2. **Run:** the only line the user edits — `python -m hallu_hierarchy.run --config configs/<file>.yaml --stages <list>`.
3. **Report:** print `REPORT.md` (so it is the last text in the log) and display `DASHBOARD.png`.
4. **Save:** outputs saved for use as a Kaggle Dataset by later stages.

**`/kaggle/working/` contains only:**

| File | Purpose |
|---|---|
| `REPORT.md` | The report (§7). The only file the user needs to read or send |
| `DASHBOARD.png` | All key plots in one image. Optional to check |
| `REVIEW.csv` | Only when the report says action is needed |
| `outputs/` | Everything else. The user never needs to open it |

Other rules:
- `--stages` lets any stage rerun alone from saved outputs (e.g. `--stages correctness`).
- Generation and M2 **checkpoint every 100 questions** and **resume** from the last checkpoint.
- Every run writes `outputs/<run>/meta.json`: git commit hash, model ID, dataset, seed, full config, timestamp.

## 6. Layout

```
hallu_hierarchy/
  CLAUDE.md
  kaggle/run.ipynb   # fixed 4-cell notebook
  configs/           # one yaml per model × dataset; refusal_patterns.yaml; generic_aliases.yaml
  overrides/         # manual label decisions, committed (§3)
  data/              # loaders, alias handling, subject-based splits
  generation/        # greedy + samples, hidden-state / attention caching, checkpointing
  correctness/       # strict/relaxed matching, alias cleaning, abstain, flags, overrides
  labeling/          # co-occurrence, AH/UH assignment
  entropy/           # token, sequence, semantic, layer-wise, attention, combination features
  probes/            # SEP training per level / layer / position
  classifier/        # two-stage + flat classifiers
  eval/              # metrics, bootstrap CIs, baselines, per-dataset / per-relation tables
  reporting/         # REPORT.md, automatic checks, DASHBOARD.png (§7)
  run.py
  requirements.txt
```

## 7. The report (every run)

Goal: the user checks **one status line**, and sends Claude **one file**. Nobody hunts through files.

`REPORT.md` is written to `/kaggle/working/` and printed as the **last thing in the log**, between clear markers. Structure:

```
==================== REPORT START ====================
STATUS: ✅ PASS                      (or ⚠️ WARN / ❌ FAIL)
Milestone: M1.5 | qwen2.5-3b | popqa | n=200 | stages: correctness | commit a1b2c3d

CHECKS
✅ no empty predictions
✅ no subject leakage across splits
✅ all checkpoints complete
⚠️ accuracy 12% (expected 15–25%)

KEY NUMBERS
  correct   36 (18.0%) | wrong 149 (74.5%) | abstain 14 (7.0%) | excluded 1
  changed vs previous run: 14 wrong → abstain, 2 wrong → correct (overrides)
  aliases dropped as generic: 3
  accuracy by relation (top 5 by n)
    occupation   n=31  12.9%  ███
    author       n=24  25.0%  █████

ACTION NEEDED                         (section only present when needed)
  7 rows need a decision → listed below (also in REVIEW.csv)
  qid | question | pred | aliases | strict | relaxed

FOR CLAUDE (paste everything between the markers)
  compact plain-text block, ≤ 60 lines: milestone, commit, config summary,
  all counts and metrics with CIs, all warnings, run time
==================== REPORT END ======================
```

Rules:
- **STATUS** is ❌ FAIL if any hard check fails, ⚠️ WARN if any soft check is outside its expected range, else ✅ PASS.
- Every stage defines its own checks with expected ranges, e.g.:
  - generation: no empty predictions, all checkpoints complete, no subject leakage
  - correctness: abstain rate < 15%, accuracy within expected range per dataset
  - entropy: no NaN/inf features, every feature has non-zero variance
  - probes/classifier: AUROC > 0.5, no class with < 30 test examples (warn)
- Each warning line says what is wrong in plain words.
- Plain-text tables and unicode bars only; no extra dependencies. Percentages next to every count; always show `n`.
- Per stage, the key numbers are:
  - **generation:** questions done, time per question, mean answer tokens, % finished cleanly
  - **correctness:** outcome counts strict vs relaxed, changes vs previous run, aliases dropped, review rows, flag counts, accuracy by relation
  - **entropy:** mean ± std of each feature per label; single-feature AUROC (correct vs wrong) sorted; correlation matrix of entropy levels
  - **labeling:** FA/AH/UH/abstain/excluded counts for each co-occurrence cutoff
  - **probes:** AUROC by layer × position, best layer marked
  - **classifier/eval:** AUROC / AUPRC with 95% CI and ECE per stage and class; comparison table against every baseline with the best marked; confusion matrix
- `DASHBOARD.png`: one image combining the stage's key plots (feature histograms per label, ROC curves vs baselines, probe AUROC vs layer, reliability diagram). Optional for the user; every important number is also in the text report.
- `summary.json` with the same numbers goes to `outputs/<run>/`.

## 8. Build order — one milestone at a time, test before moving on

1. **M1 (done):** data + generation + correctness, 200 PopQA questions.
2. **M1.5:** §3 correctness changes (alias cleaning, abstain, strict/relaxed, overrides, flags) + §5 notebook + §7 report.
   Rerun `--stages correctness` on the saved 200.
3. **Full generation:** all 14,267 PopQA questions (`--stages generation,correctness`).
4. **M2:** `entropy/` all features. Develop and check on a 200-question subset first, then run on everything.
5. **M3:** AH/UH labels with the Wikipedia proxy; manual check of ~100 labels; review flags.
6. **M4:** probes.
7. **M5:** classifier + eval with all baselines and bootstrap CIs.
8. **M6:** OLMo + infini-gram proxy validation; TriviaQA and NQ-Open; pooled and cross-dataset results.

## 9. Open decisions (defaults in place; revisit with real data)

- **Judge for review rows at scale:** NLI vs LLM judge — decide after seeing the review count on full PopQA.
- **AH/UH co-occurrence cutoff:** configurable; report 2–3 cutoffs plus manual-check agreement.
- **Automatic ambiguity detection:** Wikidata homonym check — decide at M3 whether the heuristic flag is enough.

## 10. Coding conventions

- Python 3.10+, PyTorch + `transformers`. Type hints, small functions, no notebook-only logic.
- Every stage is config-driven and idempotent: reads inputs from disk, writes outputs to disk.
- Fixed seeds everywhere, logged in `meta.json`.
- `.gitignore` covers `outputs/`, `*.npy`, `*.pt`, checkpoints.

## 11. How Claude Code works with the user

- **After writing code, always end with run instructions:**
  1. the exact text for notebook cell 2
  2. the expected result (e.g. "STATUS should be PASS, accuracy about 18%, ~14 abstain")
  3. how long the run should take
- **Label decisions:** when the user gives decisions (e.g. "Avatar = correct, Atlanta = gold_noise"), write them to the overrides file and commit.
- **At the end of each milestone:** update §0 with what was done, key numbers, and what is next. Commit.
- Never ask the user to find files inside Kaggle; everything needed is in `REPORT.md`.
