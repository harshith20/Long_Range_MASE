# CLAUDE.md — hallu_hierarchy

This folder is a self-contained research project inside a larger repo. Work only inside
`hallu_hierarchy/` unless told otherwise. Do not modify code outside this folder.

## 0. Current status (Claude Code updates this at the end of every milestone)

- **M1 / M1.5 done.** Correctness pipeline, report and Kaggle notebook work.
- **Full PopQA generation done** (commit 2ab927f): 14,267 questions, 143/143 checkpoints, 0.74 s/question
  (~2.9 h), no empty answers, subject located in 100% of prompts. Outputs saved as a Kaggle Dataset.
  - Strict accuracy 13.6% (1,942 correct), wrong 80.8%, abstain 5.6%. Consistent across splits.
  - Old ambiguity heuristic flagged 48.8% and set aside ~5,800 rows → replaced in §3 (flag no longer excludes).
  - 261 strict/relaxed disagreements → now resolved by rule (§3), no manual review.
  - **Relation confounder confirmed:** ~74% of correct answers come from closed-set relations, which are ~31% of questions.
- **Kaggle CLI automation (§12) done**; first CLI runs on kernel `harshithswamy20/hallu-hierarchy-m1-5`.
- **M1.6 done** (E04, commit 0155396): correct 2,047 (14.3%) | wrong 79.0% | abstain 5.5% | partial 1.2%.
  Closed-set 35.3% vs open-set 5.1% accuracy. 92 of 165 partials were religion ("Catholic" vs "catholic church")
  → `church`, `religion` added as head words; needs a correctness-only rerun (M1.6b).
- Experiment write-ups: `experiments/` (§13), E01–E04 so far.
- **M1.6b done** (E05, commit 06bf86f, WARN only for `capital of` refusals — real model refusals, not a rule bug):
  correct 2,128 (14.9%), partial 0.6%. Only 5/16 relations had ≥ 30 correct test examples with one test split,
  all closed-set → **evaluation switched to 5-fold CV grouped by subject** (§3 Metrics). Popularity is U-shaped
  (lowest decile 20% vs 7–11% mid) — possible lucky guesses in closed-set relations.
- **M1.6 complete** (E06 audit, E07 labels, commit 6de0783): correct 2,128 (14.9%) | wrong 11,114 | abstain 939
  (6.6%, incl. "not specified") | partial 86. 5-fold subject-grouped folds in the labels; 13/16 relations usable
  within-relation (7/9 open-set). Labels are final for M2/M3. Note: the labels with folds exist only in the
  correctness output of kernel v6 — later runs either rerun `correctness` first (10 s) or attach that output.
- **Subject-state stage done** (commit 2100edc, run 2026-11-18, STATUS ✅ PASS): 14,267/14,267 questions,
  143/143 checkpoints, 0.057 s/question (~15 min); subject found 14,264/14,267 (3 missing); prompts and
  subject positions identical to generation; last-prompt-token state matches generation (max rel. diff 0.0019).
  States `[14267, 10, 2048]` fp16 at hidden-state indices `[0, 4, …, 32, 36]`.
  - **Index 0 is the embedding layer** (HF `hidden_states[0]`), not a transformer block.
  - **Index 36 already has the final layer norm applied** (verified in E10: the lens at index 36 reproduces the
    real first-token distribution, 99.77% top-1, median |ΔH| 0.0000), so `logit_lens` is correct.
- **Labels regenerated** (E09, kernel v8, commit 8992b47): identical to E07. Final labels with folds are in the
  output of kernel v6 and v8.
- **Where the data lives (Kaggle):** saving an output as a *new version* of a dataset **replaces** its files.
  - `hallu-hierarchy-m-1-5` v3 (latest) = generations + data + pre-M1.6 labels (= v1). v2 = subject states only.
  - `hallu-hierarchy-subject-states` = subject states (uploaded by the user, 2026-11-19).
  - Labels with folds: kernel v8 output (not saved as a dataset; entropy runs regenerate them with
    `--stages correctness,entropy`, ~10 s).
  - `kernel-metadata.json` attaches `hallu-hierarchy-m-1-5` and `hallu-hierarchy-subject-states`.
- **M2 done** (E11 subset, E12 full, kernel v11, STATUS ⚠️ WARN only for `capital of` refusals): within-relation
  AUROC tok_max 0.797, sem_entropy 0.781, seq_nll 0.766 (pooled ~0.90 → H5 gap 0.11–0.15 for every feature);
  subject attention ≈0.68 within-relation (flipped); final-layer levels are one dimension (VIF up to 24);
  discrete SE ≥ weighted SE → SEP target = discrete SE. LR feature set still to be decided before M5.
- **Local CPU runs (from 2026-11-21):** CPU-only stages (correctness, labeling) can run on the user's machine:
  `hallu_hierarchy/.venv/Scripts/python.exe -m hallu_hierarchy.run --config configs/qwen3b_popqa_full.yaml --stages <s>`
  from the repo root. Inputs in `hallu_hierarchy/outputs/qwen2.5-3b_popqa_full/` (gitignored): `data/questions.jsonl`
  and `generation/generations.jsonl` downloaded from `hallu-hierarchy-m-1-5`; `correctness/labels.jsonl` regenerated
  locally (identical to kernel v8: 2,128 / 11,114 / 939 / 86, same folds). The report goes to `hallu_hierarchy/outputs/REPORT.md`
  and is copied to `reports/<date>_<run>_<stages>_local-E<NN>.md`. The .venv now also has numpy, pytest, matplotlib.
- **M3 in progress** (labeling stage, commit 61bce02). **E13** (200 wrong + 50 correct, local): proxy valid (gold answer
  on the subject page for 96.5% of wrong rows); AH 8.5% of wrong under the page rule, almost all closed-set (open 2%);
  `disambig_exists` flags 61% of subjects and removes genuine AH → recommended `answer_on_disambig` (user decision, §9).
  Wikipedia search is rate-limited (~10 s/request) → counts only on samples; the full run uses the page rule.
  Threat: AH vs FA on PopQA ≈ closed vs open relations → TriviaQA / EntityQuestions more important.
- **E14** (all rows, page rule, commit 106722b, PASS): gold on page 97%; AH 947 before the ambiguity check →
  740 (`answer_on_disambig`) / 510 (`disambig_exists`, current default); FA 2,128, UH 10,094, unscored 72.
  Within-relation AH vs FA (≥ 30 each) possible for genre, capital, capital of, religion, place of birth,
  occupation, father. Creators of works (director …) have 1–3% AH. M3 labels: `outputs/<run>/labeling/labels_m3.jsonl`
  (`label_m3`, plus `label_m3_by_rule`, `disambig_exists`, `answer_on_disambig` for switching rules).
- **Ambiguity rule decided:** `answer_on_disambig` → final labels FA 2,128 · AH 740 (closed 415, open 325) · UH 10,094 ·
  ambiguous 207 (E14 update). H9 added (fact frequency, M6).
- **Next (in order):** user fills `overrides/m3_manual_popqa_qwen2.5-3b.csv`
  (`manual_label`) → E15 offline rerun with agreement → M3 done → M4.
  TriviaQA generation + EntityQuestions pilot can run on Kaggle in parallel (GPU, separate session).

## 1. What this project is

Thesis work: detect LLM hallucinations with **semantic entropy probes (SEPs)** and a
**multi-level entropy hierarchy**, using the entropy levels as a *joint feature vector* for a
**learned classifier boundary** instead of a hand-set threshold.

| Label | Meaning |
|---|---|
| **FA** — factual association | Model's answer matches the gold answer |
| **AH** — associated hallucination | Wrong, but the wrong answer is strongly linked to the subject (e.g. "Where was Einstein born?" → "Munich") |
| **UH** — unassociated hallucination | Wrong, and the answer has little or no link to the subject (→ "Lisbon") |
| **abstain** | Refusal ("Not enough information", "Unknown", …). Not a hallucination |
| **partial** | Correct but less specific than gold, or too vague to judge ("Singer" vs "singer-songwriter", "Fiction" vs "science fiction film"). Not a hallucination |

`abstain` and `partial` are excluded from FA/AH/UH training and always counted in reports.

### Hypotheses

| ID | Priority | Claim | Test |
|---|---|---|---|
| **H1** | Core | The entropy hierarchy separates UH from FA | AUROC UH vs FA vs all baselines |
| **H1b** | Core | Single-pass probes keep most of the performance of full sampling (the edge claim) | Probe-based vs full-sampling classifier: AUROC and inference time side by side |
| **H2** | Core | Entropy alone fails on AH vs FA (AH is confident and consistent) | AH vs FA AUROC with entropy features only; expect ≈ 0.5 |
| **H3** | Core | Non-confidence internal signals ($\ell^*$, layer-wise entropy, attention to subject) partly recover AH vs FA | AH vs FA AUROC: entropy only vs + layer-wise + attention |
| **H4** | Recommended | Subject familiarity (probe on subject-token state) separates UH but not AH: AH = familiar subject + unknown fact | Familiarity probe predicting `s_pop`; compare its scores across FA / AH / UH |
| **H5** | Core | Detection depends on relation type; pooled AUROC is inflated by it | Within-relation vs pooled AUROC; open vs closed groups; leave-one-relation-out |
| **H6** | Optional | AH answers change more under paraphrasing than FA answers | 3 paraphrases per question; answer agreement per class (analysis only, not an edge feature) |
| **H7** | Optional | Many errors on ambiguous subjects are not hallucinations | Regenerate flagged subset with full Wikipedia titles; compare accuracy |
| **H8** | Recommended | The detector transfers across datasets | Train PopQA → test TriviaQA, and the reverse |
| **H9** | Recommended | Errors come from rare facts: accuracy rises with how often the subject and the *true* answer co-occur in the training data (fact frequency), and among wrong answers the AH share depends on it (long-tail facts are lost under limited capacity) | OLMo (M6): infini-gram counts of subject + gold answer in its pretraining corpus; accuracy and FA/AH/UH shares per frequency bin, within relation. Qwen: `s_pop` and Wikipedia counts as weaker proxies (analysis only, never a feature) |

Thesis story: does it work (H1, H1b) → where it fails (H2) → can the failure be fixed (H3) →
why it fails (H4, H6, H9) → are the results trustworthy (H5, H7, H8).

A clearly documented negative result on H2 (or H3) is acceptable. Do not tune things to hide it.

## 2. Non-negotiable rules

1. **Labels never become features.** FA/AH/UH come from gold answers + co-occurrence. Semantic entropy is a feature, never a label.
2. **Split by subject entity, not randomly.** The same subject must never appear in both train and test.
   Evaluation uses **5-fold cross-validation grouped by subject** (`fold` column in the labels).
3. **Inference must stay cheap (edge setting).** Sampling + NLI is allowed only to *create training targets* and for baselines. At inference time, only single-pass signals: probe outputs, greedy-pass token entropy, layer-wise features, attention features, subject-familiarity probe, classifier.
4. **Entropy levels are correlated, not independent.** Report the correlation matrix and VIF between them.
5. **Labels are per model.** Never share labels across models.
6. **Never commit outputs.** Hidden states and generations go to `/kaggle/working/outputs/` (or `outputs/` locally, gitignored).
7. **Never judge correctness by embedding similarity.** "Ulm" and "Munich" are close in embedding space, so it would mark associated hallucinations as correct.
8. **Relaxed matching never auto-promotes wrong → correct.** Rows that match only under relaxed matching become `partial` (§3).
9. **`gold_noise` and `partial` rows are excluded from FA/AH/UH training.** The `ambiguous_subject` flag does **not** exclude anything by itself; it is a column used for the M3 check and for sensitivity analysis.
10. **The dataset name and relation type are never classifier features.** Relation type is used only for analysis (H5).
11. **Pooled AUROC is never reported alone.** Always next to within-relation AUROC and open/closed group results.

## 3. Pipeline

```
QA data → generation (greedy + N samples, cache hidden states + attention)   [done for PopQA]
        → subject-state stage (prompt-only forward pass, subject-token hidden states)   [H4]
        ├→ correctness: correct / wrong / abstain / partial (+ flags, + overrides)
        │     └→ wrong only: AH / UH via co-occurrence (+ ambiguity check on AH rows)
        └→ entropy hierarchy: token, sequence, semantic, layer-wise, attention
              → probes on hidden states (entropy levels + subject familiarity)
                    → boundary classifier (features → FA / AH / UH)
                          → evaluation vs baselines; per dataset, per relation, open/closed
```

### Generation (done for PopQA)
- Short-form QA prompt, answer in a few words, using each model's chat template.
- 1 greedy answer (temperature 0) + N = 10 samples (temperature 1.0), max ~32 new tokens.
- Hidden states for the **greedy** pass at **TBG** (last prompt token) and **SLT** (second-last answer token), every 4th layer + last, fp16.
- Per-token entropy of the greedy answer; log-probs of all samples.
- Attention row of the last prompt token at selected middle/late layers (all heads); subject token positions via tokenizer offsets.

### Subject-state stage (`--stages subject_states`, for H4)
- One forward pass over the **prompt only** (no generation), reusing the same prompt and tokenizer.
- Cache the hidden state at the **subject's last token**, same layers as above, fp16.
- Much cheaper than generation. Checkpoint every 100 questions like generation.
- Must produce identical prompts/token positions to the generation stage (assert this on a sample).

### Correctness (M1.6 = rerun with the changes below)
Runs on saved generations; no GPU. Applied to the greedy answer **and** to every sample.

1. **Normalize** both strings: lowercase, strip punctuation and articles (a/an/the).
2. **Alias cleaning:** drop single generic-word aliases (`configs/generic_aliases.yaml`, e.g. "church", "film", "music", "movie", "series"). Log every dropped alias.
3. **Abstain:** refusal pattern list in `configs/refusal_patterns.yaml`. Seed: "not enough information", "not applicable", "unknown", "unnamed", "none", "nonegiven", "none given", "is not", "various".
   **Exception:** "none" is *not* a refusal for the `religion` relation (it can mean no religion). Patterns support per-relation exemptions. Log every match.
4. **`match_strict`:** a normalized gold alias appears in the prediction as a whole-word span.
5. **Head-word stripping:** also build a stripped copy of each alias with trailing generic head words removed
   (`film`, `movie`, `series`, `tv series`, `video game`, `game`, `genre`, `music`, `literature`; list in config).
   A strict match against a stripped alias counts as `match_strict`. Example: "Comedy" vs "comedy film" → correct.
6. **`match_relaxed`:** prediction is a whole-word span of an alias, with length ≥ 2 words or ≥ 6 characters.
7. **`label_auto`:**
   - `abstain` if a refusal pattern matches
   - `correct` if `match_strict`
   - `partial` if only `match_relaxed` (e.g. "Singer" vs "singer-songwriter", "New York City" vs "Queens", "Fiction" vs "science fiction film")
   - `wrong` otherwise
8. **Overrides:** `overrides/<dataset>_<model>.csv` (`qid, label, gold_noise, ambiguous_subject, note`). Final `label` = override if present, else `label_auto`. Claude Code writes decisions the user gives and commits them. Nobody edits files inside Kaggle. Existing overrides:

    | qid | label | gold_noise | ambiguous_subject | note |
    |---|---|---|---|---|
    | popqa-1919197 | correct | 0 | 0 | Avatar → "Science fiction"; short form of gold |
    | popqa-1823075 | wrong | 1 | 0 | Atlanta → "Georgia" is right; PopQA gold "Fulton County" is wrong. Excluded |
    | popqa-384830 | correct | 0 | 0 | Edmund A. Walsh → "Catholic"; short form of "Catholic Church" |

9. **No manual review list by default.** `partial` replaces it. The report lists counts of `partial` per relation and 10 random examples.
10. **`ambiguous_subject` flag:** keep the single-word/generic-title heuristic as a column only. It no longer excludes rows. The real check happens in M3 on AH-labelled rows.
11. **`gold_noise`:** set by override only.
12. Report changes vs the previous run (counts moved between labels).

### Relation groups (for H5 and all reporting)
Defined in `configs/relation_groups.yaml`:
- **closed-set** (few plausible answers, guessable): country, capital, capital of, sport, genre, religion, color
- **open-set** (person or entity names, recall needed): director, screenwriter, producer, author, composer, father, mother, place of birth, occupation

### AH vs UH labeling (M3; wrong answers only, excluding `gold_noise` and `partial`)
- Co-occurrence of subject and the model's wrong answer. High → AH, low → UH.
  - OLMo: counts in OLMo's open pretraining corpus via infini-gram (check the index matches the model's training data version).
  - Qwen: Wikipedia proxy (answer appears on the subject's page / count of pages mentioning both).
- The cutoff is a parameter. Report results for 2–3 cutoffs.
- **Ambiguity check on AH rows only:** does a Wikipedia disambiguation page exist for the subject, or is the model's answer true for another entity with the same name? If so → `ambiguous_confirmed`, excluded from AH. Report AH counts before and after.
- Manual check of ~100 labels; report agreement.
- **Proxy validation:** on OLMo, label with both corpus counts and the Wikipedia proxy; report agreement.
- Sensitivity analysis: all main results with and without rows carrying the heuristic `ambiguous_subject` flag.

### Entropy hierarchy (features, M2)
- **Token:** mean and max per-token entropy of the greedy answer. Mixes lexical, syntactic and factual uncertainty.
- **Sequence:** mean length-normalized NLL across the N samples. Whole-answer uncertainty; still inflated by paraphrases.
- **Semantic:** cluster the N samples by bidirectional entailment (DeBERTa-large-MNLI), entropy over cluster frequencies. Meaning-level only. Abstaining samples form **their own cluster**. Also record `abstain_frac` among samples (training-time only).
- **Layer-wise (logit lens):** at the **last prompt token**, $p^{(\ell)} = \text{softmax}(W_U \cdot \text{LN}(h^{(\ell)}))$ for each cached layer. Record entropy at middle and late layers (ignore early layers) and $\ell^*$ = first layer where the final answer's first token becomes top-1.
- **Attention to subject:** $q$ = last prompt token, $S$ = subject token positions, $a^{(\ell,h)} = \sum_{j \in S} A^{(\ell,h)}_{q,j}$. Mean and max over heads, at 3 middle/late layers.
- **Combination features:**
  - Logistic regression: **no difference features**. Use the ratio $H_{\text{sem}} / (H_{\text{tok}} + \epsilon)$.
  - Gradient boosting: differences of z-scored levels are allowed.
- NLI clustering is the slow part: checkpoint every 100 questions and resume.

### Probes (M4)
- Logistic regression probes, one per entropy level, as in the SEP setup (binarize the target entropy at a training-set split point; this threshold only creates probe targets). Per layer and per position (TBG vs SLT).
- **Subject-familiarity probe (H4):** on subject-token states, predicting high vs low `s_pop` (split at the training median). Per layer.
- Probes stay linear (high-dimensional input, limited data; linear readability is part of the claim).

### Classifier (M5)
- **Two-stage:** (1) correct vs hallucinated, (2) AH vs UH. Plus a flat 3-class version.
- Logistic regression first, then gradient boosting. Optional explicit interaction terms (e.g. $H_{\text{sem}} \times$ open-set) for logistic regression in the H5 analysis only.
- `class_weight="balanced"` (or equivalent) everywhere.
- Inference feature vector uses probe outputs in place of the sampled $H_{\text{sem}}$, $H_{\text{seq}}$.
- **Feature-set ablations** for H2/H3: entropy only → + layer-wise → + attention → + subject familiarity.
- **Inference cost (H1b):** log per-question time for the single-pass detector vs full sampling + NLI.

### Baselines (required)
- Semantic entropy (full sampling), SEP alone, token entropy alone
- **Direct correctness probe** (linear probe predicting correct/wrong from hidden states). If the hierarchy does not beat this, say so.
- **Relation-only baseline** (H5): a classifier that sees only the relation type. Anything near its score is detecting question type, not hallucination.
- The previous single-threshold entropy-hierarchy method from earlier thesis work

### Metrics
- **Evaluation protocol: 5-fold cross-validation grouped by subject** (decided after E05: a single 20% test split
  leaves no open-set relation with ≥ 30 correct test examples). Folds are assigned in the correctness stage
  (`fold` column, deterministic from the seed). For each outer fold: train on 4 folds, test on the 5th.
  Probe layer, thresholds and hyperparameters are chosen on an **inner subject-grouped split of the 4 training
  folds only** — never on the test fold. Every question gets exactly one out-of-fold score.
- Metrics are computed on the **pooled out-of-fold scores**; also report mean ± std over the 5 folds.
- AUROC and AUPRC per class (one-vs-rest) and per stage; ECE.
- **Bootstrap 95% CIs** on every AUROC/AUPRC (1,000 resamples of the pooled out-of-fold scores, resampling by subject).
- **Primary tables:** within-relation AUROC (macro average over relations with ≥ 30 out-of-fold examples per class), open-set vs closed-set results, pooled AUROC alongside.
- Leave-one-relation-out results.
- Per-dataset, pooled and cross-dataset (H8) tables.

### Optional stages
- **H6 paraphrase stage** (`--stages paraphrase`): 3 paraphrase templates per relation (in config), greedy answer only; answer agreement per class. Analysis only.
- **H7 disambiguation stage** (`--stages disambig`): regenerate greedy answers for flagged subjects with the full Wikipedia title in the question; compare accuracy.

## 4. Models and datasets (Kaggle free tier: one T4, 16 GB)

Verify exact Hugging Face IDs before use.

| Role | Model |
|---|---|
| Main | `Qwen/Qwen2.5-3B-Instruct` (fp16, ~6 GB) |
| Open-pretraining-data check | OLMo-2 1B Instruct (`allenai/OLMo-2-0425-1B-Instruct`) |
| Semantic clustering | `microsoft/deberta-large-mnli` |

No 7B models: on a T4 they need 4-bit quantization, which alters the hidden states being probed.

| Order | Dataset | Role |
|---|---|---|
| 1 | **PopQA** (all 14,267) | Long-tail entities, template questions; UH common; has `s_pop`, relation `prop`, Wikipedia titles. Done |
| 2 | **TriviaQA** (rc.nocontext, validation) | Human-written, specific, rarely ambiguous; more correct answers incl. people; AH expected more often. Run **before OLMo** |
| 3 | NQ-Open | Optional realism check; a known large share of its questions are ambiguous |
| pilot | **EntityQuestions** (200-question pilot first) | Wikidata templates like PopQA, with relation and subject fields; many person (open-set) relations, more popular subjects → candidate source of open-set FA and AH. Full run only if the pilot shows a useful correct/wrong mix |

**TriviaQA has no subject field.** Subject span is needed for attention features, H4 and co-occurrence.
Default: use the question's linked entity titles from TriviaQA's entity pages and locate them in the question text (fuzzy match).
Questions where no subject can be located get `subject_found = False`, are excluded from subject-dependent features and analyses, and are counted in the report.

**Pooling and reporting:** all rows carry a `dataset` column. Classifiers are trained per dataset **and** pooled.
Datasets differ in difficulty, so class balance differs by dataset: never balance by dropping evaluation data
(use class weights), and watch for the dataset acting as a shortcut (question style ≈ label) — per-dataset
results and cross-dataset transfer (H8) are the guard.
Always report per dataset; never only pooled numbers. Cross-dataset transfer for H8.

## 5. Kaggle workflow

Runs are started from the user's machine with the **Kaggle CLI**, by Claude Code, through the guarded
script `scripts/kaggle_run.sh` (§12). The browser (Save & Run All) remains a fallback and uses the same notebook.
Results are read from `REPORT.md`, downloaded into `reports/`.

**Fixed notebook** `kaggle/run.ipynb`, 4 cells, identical every run:
1. **Setup:** clone the repo (token from Kaggle Secrets), install requirements, HF token from Kaggle Secrets, attach previous outputs Dataset.
2. **Run:** the only line the user edits — `python -m hallu_hierarchy.run --config configs/<file>.yaml --stages <list>`.
3. **Report:** print `REPORT.md` (last text in the log) and display `DASHBOARD.png`.
4. **Save:** outputs saved for use as a Kaggle Dataset by later stages.

**`/kaggle/working/` contains only:**

| File | Purpose |
|---|---|
| `REPORT.md` | The report (§7). The only file the user reads or sends |
| `DASHBOARD.png` | Key plots in one image. Optional |
| `REVIEW.csv` | Only when the report says action is needed |
| `outputs/` | Everything else. The user never opens it |

- `--stages` reruns any stage alone from saved outputs.
- Generation, subject-state and entropy stages checkpoint every 100 questions and resume.
- Every run writes `outputs/<run>/meta.json`: git commit hash, model ID, dataset, seed, full config, timestamp.

## 6. Layout

```
hallu_hierarchy/
  CLAUDE.md
  kaggle/run.ipynb   # + kernel-metadata.json
  scripts/kaggle_run.sh   # guarded CLI runner (§12)
  reports/           # committed REPORT.md history (raw, one per run)
  experiments/       # one folder per experiment: why, setup, results, insights, next steps (§13)
  configs/           # model × dataset yamls; refusal_patterns, generic_aliases, head_words, relation_groups, paraphrases
  overrides/         # manual label decisions, committed
  data/              # loaders, alias handling, subject location (incl. TriviaQA), subject-based splits
  generation/        # greedy + samples, hidden states / attention, subject-state stage, checkpointing
  correctness/       # strict/relaxed matching, alias cleaning, head-word stripping, abstain, partial, flags, overrides
  labeling/          # co-occurrence, AH/UH, ambiguity check on AH rows
  entropy/           # token, sequence, semantic, layer-wise, attention, combination features
  probes/            # entropy-level probes, subject-familiarity probe
  classifier/        # two-stage + flat classifiers, ablations
  eval/              # metrics, bootstrap CIs, baselines, within-relation / group / transfer tables
  reporting/         # REPORT.md, checks, DASHBOARD.png
  run.py
  requirements.txt
```

## 7. The report (every run)

Goal: the user checks **one status line** and sends Claude **one file**.

`REPORT.md` is written to `/kaggle/working/` and printed as the **last thing in the log**, between
`REPORT START` / `REPORT END` markers:

1. **STATUS** line (✅ PASS / ⚠️ WARN / ❌ FAIL) + milestone, model, dataset, n, stages, commit.
2. **CHECKS**: one line each, ✅/⚠️/❌, with plain-words explanation for every ⚠️/❌.
3. **KEY NUMBERS**: per stage, small plain-text tables with percentages and `n`.
4. **ACTION NEEDED**: only when needed; rows printed inline (up to 20), rest in `REVIEW.csv`.
5. **FOR CLAUDE**: compact block (≤ 60 lines) with every number and setting Claude would ask about.

Checks (expected ranges):
- generation / subject-state: no empty predictions, all checkpoints complete, no subject leakage, subject found ≥ 95%
- correctness: abstain < 15%, partial < 5%, no relation abstains > 25%, accuracy within expected range per dataset, no subject in two folds
- entropy: no NaN/inf, every feature has non-zero variance
- probes/classifier: AUROC > 0.5; warn if any class has < 30 out-of-fold examples; warn if within-relation AUROC is much lower than pooled (> 0.10 gap)

Key numbers per stage:
- **generation / subject-state:** done / total, time per question, answer tokens, % finished cleanly, subject found
- **correctness:** counts for correct / wrong / abstain / partial, changes vs previous run, aliases dropped, head-word matches, flag counts, **label counts per relation and per open/closed group**
- **entropy:** mean ± std of each feature per label; single-feature AUROC (correct vs wrong) sorted, **pooled and within-relation**; correlation matrix
- **labeling:** FA/AH/UH/abstain/partial counts per cutoff and per group; AH before/after ambiguity check
- **probes:** AUROC by layer × position; familiarity probe AUROC by layer; best marked
- **classifier/eval:** AUROC / AUPRC with 95% CI and ECE per stage and class; comparison vs every baseline incl. relation-only; feature-set ablation table; within-relation and open/closed tables; inference time per question

Plain-text tables and unicode bars only. `DASHBOARD.png` combines the stage's key plots; optional for the user.
`summary.json` with the same numbers goes to `outputs/<run>/`.

## 8. Build order — one milestone at a time, test before moving on

1. **M1, M1.5, full PopQA generation:** done.
2. **M1.6:** §3 correctness changes (head-word stripping, `partial`, religion exemption, flag no longer excludes, relation groups in reports). `--stages correctness`. No GPU. Done (E04–E07, E09).
3. **Subject-state stage** (H4): `--stages subject_states` on all PopQA. Done (E08).
4. **M2:** `entropy/` all features. Develop and check on a 200-question subset first, then run on all. Done (E11, E12).
5. **TriviaQA generation** (+ subject location, correctness, subject states) — can run in a separate Kaggle session while M3–M5 are developed.
   **EntityQuestions pilot** (200 questions) in the same period; full run only if its correct/wrong mix helps open-set relations.
6. **M3:** AH/UH labels with the Wikipedia proxy; ambiguity check on AH rows; manual check of ~100.
7. **M4:** entropy-level probes + familiarity probe.
8. **M5:** classifier, ablations, all baselines, bootstrap CIs, within-relation and group tables. Answers H1, H1b, H2, H3, H4, H5.
9. **M6:** OLMo + infini-gram proxy validation; cross-dataset transfer (H8); NQ-Open optional.
10. **Optional:** H6 paraphrase stage, H7 disambiguation stage.

## 9. Open decisions (defaults in place; revisit with real data)

- **AH/UH co-occurrence cutoff:** configurable; report 2–3 cutoffs plus manual-check agreement.
- **Ambiguity check method on AH rows:** decided at M3 (user, 2026-11-21, after E14): `answer_on_disambig` — an AH row is
  `ambiguous_confirmed` only if the model's answer appears on the subject's disambiguation page. `disambig_exists`
  (any disambiguation page) removed 46% of AH incl. unambiguous subjects; it is kept as a sensitivity analysis.
- **TriviaQA subject location:** entity-page titles by default; if too many questions have no subject found, consider an NER or LLM-based extractor.
- **Relation group membership:** occupation is borderline (open vs closed); keep configurable and report both.

## 10. Coding conventions

- Python 3.10+, PyTorch + `transformers`. Type hints, small functions, no notebook-only logic.
- Every stage is config-driven and idempotent: reads inputs from disk, writes outputs to disk.
- Fixed seeds everywhere, logged in `meta.json`.
- `.gitignore` covers `outputs/`, `*.npy`, `*.pt`, checkpoints.

## 11. How Claude Code works with the user

- **Show the diff and explain the change before committing.** The user reads the code to learn it.
- **Before every run, state:** the stages, GPU on/off, expected duration, and expected result
  (e.g. "STATUS should be PASS, ~2,000 correct, partial ~1–2%"). Then run it through `scripts/kaggle_run.sh` (§12).
- **Label decisions:** write decisions the user gives to the overrides file and commit.
- **After every run:** write the experiment folder (§13) before proposing the next run.
- **At the end of each milestone:** update §0 with what was done, key numbers, and what is next. Commit.
- Never ask the user to find files inside Kaggle; everything needed is in `REPORT.md`.

## 12. Kaggle CLI automation and safety

### Flow for every run
1. Claude Code changes code → shows the diff → user agrees → commit and `git push`.
2. Claude Code states stages, GPU on/off, expected duration (§11).
3. Claude Code calls `scripts/kaggle_run.sh`. The Claude Code permission prompt shows the full command; **the user's approval of that prompt is the "yes".**
4. The script pushes the kernel, waits, downloads outputs, and copies `REPORT.md` to `reports/<YYYY-MM-DD>_<run>_<stages>.md`.
5. Claude Code reads the report, tells the user STATUS and the key numbers, writes the experiment folder (§13),
   commits both, and **stops**.

### Rules for Claude Code (must follow)
1. **Never start a run without the user's explicit yes in the current conversation.** Approval does not carry over to later runs.
2. **Never run the `generation` stage** unless the user asks for it by name in the current conversation.
3. **CPU-only stages run with the GPU off:** correctness, labeling, probes, classifier, eval, reporting.
   GPU on only for: generation, subject_states, entropy (NLI), paraphrase, disambig.
4. **One run at a time. Never chain runs automatically**, even if the next step is obvious. After a run: report and stop.
5. **Never retry a failed GPU run automatically.** Report the failure and wait.
6. **Never read, print, copy, move or commit `~/.kaggle/kaggle.json`** or any token. Never put tokens in code, configs, logs or reports.
7. **Never delete or overwrite** Kaggle datasets, kernels or kernel versions. No `kaggle ... delete` commands.
8. **Never change Kaggle account settings** or create new Kaggle datasets/kernels without asking.
9. If anything unexpected happens (unknown error, quota message, run much longer than expected): stop and tell the user.

### Guards inside `scripts/kaggle_run.sh` (enforced in code)
- Required arguments: `--stages <list>` and `--gpu on|off`. No defaults for either.
- Refuses `--gpu on` for CPU-only stages, and `--gpu off` for GPU stages (lists from rule 3, in one config file).
- Refuses `generation` unless `--allow-generation` is also passed.
- Refuses to run if the git working tree is dirty or the current commit is not pushed (Kaggle clones from GitHub, so it would run old code). Passes the commit hash to the run and checks it appears in `REPORT.md`.
- Prints a summary before pushing: stages, GPU on/off, commit, kernel id.
- Waits with `kaggle kernels status` at most every 60 s; stops waiting (without cancelling the run) after a configurable time limit and tells the user.
- Downloads outputs with `kaggle kernels output`, copies only `REPORT.md` (and `DASHBOARD.png`) into `reports/`.
- Never prints environment variables or credentials.

### Kernel setup
- `kaggle/kernel-metadata.json`: kernel id, notebook path, internet on, GPU flag set per run by the script,
  previous outputs attached as sources. **Check the current Kaggle CLI documentation** for the exact
  metadata fields (GPU type selection, attaching previous kernel outputs or datasets); do not assume.
- The notebook reads stages from a small config the script writes (not by editing cells).

### Claude Code permission settings (`.claude/settings.json` at the repo root)
Enforced by Claude Code itself. Verify the exact syntax against the current Claude Code documentation.
- **allow** (no prompt): `kaggle kernels status`, `kaggle kernels list`, `git status`, `git diff`, `git log`
- **ask** (prompt every time; the user must never choose "always allow" for these): `scripts/kaggle_run.sh`, `kaggle kernels push`, `kaggle kernels output`, `git push`
- **deny**: any `kaggle` command containing `delete`; `kaggle datasets create`; `kaggle datasets version`;
  reading `~/.kaggle/**`; `rm -rf` outside `outputs/`

### Repo hygiene
- `.gitignore`: `kaggle.json`, `.kaggle/`, `outputs/`, `*.npy`, `*.pt`, checkpoints.
- `reports/` **is** committed (small text files; history of every run's results).

## 13. Experiment log (`experiments/`)

Every run that produces results gets one folder, written by Claude Code right after the run (§12 step 5).
`reports/` keeps the raw `REPORT.md`; `experiments/` keeps the *meaning*. Keep each write-up short:
only what is useful for the thesis or for deciding the next step. No copied logs, no full tables.

```
experiments/
  README.md                         # index: one line per experiment (id, date, question, outcome)
  E04_m1-6_correctness/README.md    # E<NN>_<short-name>, numbered in run order
```

Each `README.md` (aim for ≤ 60 lines):
1. **Header:** date, commit, stages, GPU on/off, kernel version, duration, STATUS, link to the report in `reports/`.
2. **Why:** the question the run answers and which hypothesis or milestone it serves (1–3 lines).
3. **Setup:** only what differs from the previous run (config, data, code change).
4. **Results:** one small table with the numbers that matter; changes vs the previous experiment.
5. **Insights:** what the numbers mean, incl. surprises and threats to validity (bullets).
6. **Next steps:** what follows from the insights, in order, each with the reason.

Failed attempts (crash, script bug) get one line in the experiment they belong to, not their own folder.
A rerun after a small change gets its own short folder that links back and reports only the delta
(e.g. E05 reruns E04 with two more head words).
