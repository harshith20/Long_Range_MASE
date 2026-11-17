# CLAUDE.md — hallu_hierarchy

This folder is a self-contained research project inside a larger repo. Work only inside
`hallu_hierarchy/` unless told otherwise. Do not modify code outside this folder.

## 0. Current status (update this section after each milestone)

- **M1 done:** Qwen2.5-3B-Instruct on 200 PopQA questions ran end to end on Kaggle. One question per
  subject (no leakage), all greedy answers ended cleanly, about 4 tokens per answer. Accuracy about 17%.
- **Found in M1:** (a) strict alias matching misses some correct short forms; (b) some PopQA questions
  are ambiguous titles or have wrong gold answers; (c) about 7% of answers are refusals, previously counted as wrong.
- **M1.5 code done (not yet run on Kaggle):** `correctness/` (abstain via `configs/refusal_patterns.yaml`,
  strict/relaxed, flags, review list, overrides in `review/<run_name>_overrides.csv`, old vs new labels)
  + `reporting/` summary blocks for generation and correctness. CPU tests pass.
- **Next:** run `kaggle/m1_5_correctness.ipynb` (`--stages correctness --input-run-dir <M1 outputs>`),
  audit `abstain_log.csv`, review `review/disagreements.csv` + `wrong.csv`, commit overrides, then M2.
- **Known gap:** generation still saves head-averaged attention mass per layer, not the full attention row
  of the last prompt token (§3 Generation). Fix before the next GPU run (M2 needs it for max-over-heads).

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
3. **Inference must stay cheap (edge setting).** Sampling + NLI is allowed only to *create training targets*. At inference time, only single-pass signals: probe outputs, greedy-pass token entropy, layer-wise features, attention features, classifier.
4. **Entropy levels are correlated, not independent.** Report the correlation matrix and VIF between them.
5. **Labels are per model.** Never share labels across models.
6. **Never commit outputs.** Hidden states and generations go to `/kaggle/working/` (or `outputs/` locally, gitignored).
7. **Never judge correctness by embedding similarity.** "Ulm" and "Munich" are close in embedding space, so it would mark associated hallucinations as correct.
8. **Relaxed matching never auto-promotes wrong → correct.** It only flags rows for review (§3).
9. **Flagged rows (`ambiguous_subject`, `gold_noise`) are excluded from AH/UH training** and counted in every report.

## 3. Pipeline

```
QA data → generation (greedy + N samples, cache hidden states + attention)
        ├→ correctness: correct / wrong / abstain (+ flags)
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
- Save the attention row of the last prompt token at selected middle/late layers (all heads), plus the token positions of the subject in the prompt (find them via tokenizer offsets).

### Correctness (M1.5)
Runs on saved generations; no GPU needed. Applied to the greedy answer **and** to every sample.

1. **Normalize** both strings: lowercase, strip punctuation and articles (a/an/the).
2. **Abstain:** if the answer matches the refusal pattern list → `abstain`. Seed patterns from M1:
   "not enough information", "not applicable", "unknown", "unnamed", "none", "nonegiven",
   "is not a", "various". Keep the list in a config file and log every match so it can be audited.
3. **`match_strict`:** a normalized gold alias appears in the prediction as a whole-word span.
4. **`match_relaxed`:** also true when the prediction is a whole-word span of an alias, with
   prediction length ≥ 2 words or ≥ 6 characters (so "Drama" does not match "legal drama").
5. **`label`** = `correct` if `match_strict`, else `wrong` (unless abstain). **Relaxed never changes `label`.**
6. **Review list:** rows where strict and relaxed disagree are written to `review/disagreements.csv`.
   A manual decision column there can override `label`; overrides are stored in a separate file, not by editing outputs.
   At scale, an NLI or LLM judge replaces manual review for these rows only.
7. **Flags:**
   - `ambiguous_subject`: heuristic first pass (single-word or generic subject titles, e.g. *Emma*, *Messiah*, *Boston*, *Avatar*, *Pygmalion*). Later: a Wikidata check of whether the model's answer is true for another entity with the same name.
   - `gold_noise`: the gold answer is itself wrong (e.g. "What is Atlanta the capital of?" → gold "Fulton County"). Set manually during review.
8. Keep old and new labels side by side and print what changed (see §7).

### AH vs UH labeling (M3, wrong answers only, unflagged rows only)
- Co-occurrence of subject and the model's wrong answer. High → AH, low → UH.
  - OLMo: counts in OLMo's open pretraining corpus via infini-gram (check that the index matches the model's training data version).
  - Qwen: Wikipedia proxy (answer appears on the subject's page / count of pages mentioning both).
- The cutoff is a parameter. Report results for 2–3 cutoffs.
- Manual check of ~100 labels; report agreement.
- **Proxy validation:** on OLMo, label with both corpus counts and the Wikipedia proxy; report agreement.

### Entropy hierarchy (features)
- **Token:** mean and max per-token entropy of the greedy answer. Captures lexical, syntactic *and* factual uncertainty mixed together.
- **Sequence:** mean length-normalized NLL across the N samples. Whole-answer uncertainty; still inflated by paraphrases.
- **Semantic:** cluster the N samples by bidirectional entailment (DeBERTa-large-MNLI), entropy over cluster frequencies. Meaning-level uncertainty only. Abstaining samples form **their own cluster**. Also record `abstain_frac` among samples (training-time only).
- **Layer-wise (logit lens):** at the **last prompt token**, apply the final layer norm and unembedding to the hidden state of each cached layer: $p^{(\ell)} = \text{softmax}(W_U \cdot \text{LN}(h^{(\ell)}))$. Record the entropy at middle and late layers (ignore early layers; the logit lens is unreliable there) and $\ell^*$ = first layer where the final answer's first token becomes top-1.
- **Attention to subject:** with $q$ = last prompt token and $S$ = subject token positions,
  $a^{(\ell,h)} = \sum_{j \in S} A^{(\ell,h)}_{q,j}$. Use both the **mean** and the **max** over heads, at 3 middle/late layers.
- **Combination features:**
  - For logistic regression: **no difference features** (it can learn linear combinations itself). Use the ratio $H_{\text{sem}} / (H_{\text{tok}} + \epsilon)$.
  - For gradient boosting: differences of z-scored levels are allowed.

### Probes
- Logistic regression probes on cached hidden states, one per entropy level, as in the SEP setup (binarize the target entropy at a training-set split point; this threshold only creates probe targets).
- Report probe AUROC per layer and per position (TBG vs SLT).

### Classifier
- **Two-stage:** (1) correct vs hallucinated, (2) AH vs UH. Plus a flat 3-class version.
- Logistic regression first (interpretable weights), then gradient boosting.
- Inference feature vector uses probe outputs in place of the sampled $H_{\text{sem}}$, $H_{\text{seq}}$.

### Baselines (required)
- Semantic entropy (full sampling), SEP alone, token entropy alone
- **Direct correctness probe** (linear probe predicting correct/wrong from hidden states). If the hierarchy does not beat this, say so.
- The previous single-threshold entropy-hierarchy method from earlier thesis work

### Metrics
- AUROC and AUPRC per class (one-vs-rest) and per stage; ECE; per-dataset and **per-relation** tables.

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
| PopQA | Popularity score per subject (`s_pop`); relation type per question (`prop`) |
| TriviaQA (rc.nocontext) | Mostly well-known topics; expect more AH |
| NQ-Open | Real search queries, a realistic mix |

**Sampling and scale:**
- Accuracy is low (~17% on PopQA), so correct examples are scarce. Sample PopQA **stratified by `s_pop`** so popular subjects (where the model is right, and where AH happens) are well represented.
- Choose the number of questions from M1's measured time per question so that one run fits a Kaggle session with checkpointing. Target ≥ 2,000 questions per dataset once M2 works.
- Report accuracy and label counts **per relation type**; relations dominated by ambiguity or gold noise may be dropped, and any drop is documented.

## 5. Kaggle workflow

- Code lives in this repo. The Kaggle notebook is thin: clone → install → `python -m hallu_hierarchy.run --config configs/<file>.yaml --stages <list>`.
- `--stages` lets any stage rerun alone from saved outputs (e.g. `--stages correctness`).
- Secrets (GitHub token for a private repo, HF token) come from Kaggle Secrets. Never hardcode them.
- Generation must **checkpoint every 100 questions** and **resume** from the last checkpoint.
- Save finished outputs as a Kaggle Dataset so later stages can load them without a GPU rerun.
- Every output folder gets a `meta.json`: git commit hash, model ID, dataset, seed, full config, timestamp.

## 6. Layout

```
hallu_hierarchy/
  CLAUDE.md
  configs/        # one yaml per model × dataset; refusal_patterns.yaml
  data/           # loaders, alias handling, s_pop-stratified sampling, subject-based splits
  generation/     # greedy + samples, hidden-state / attention caching, checkpointing
  correctness/    # strict/relaxed matching, abstain, flags, review list, overrides
  labeling/       # co-occurrence, AH/UH assignment
  entropy/        # token, sequence, semantic, layer-wise, attention, combination features
  probes/         # SEP training per level / layer / position
  classifier/     # two-stage + flat classifiers
  eval/           # metrics, baselines, per-dataset / per-relation tables
  reporting/      # console summaries + plots (see §7)
  run.py
  requirements.txt
```

## 7. Logging and visualization (every stage)

The Kaggle log must make results readable without opening any file.

**Console, at the end of every stage, print one summary block:**
```
================ STAGE: correctness | qwen2.5-3b | popqa | n=200 ================
Outcome counts            strict    relaxed
  correct                  34 (17.0%)  41 (20.5%)
  wrong                   152 (76.0%) 145 (72.5%)
  abstain                  14 ( 7.0%)  14 ( 7.0%)
Changed vs previous run:  14 wrong → abstain
Review list:              7 rows   → review/disagreements.csv
Flags:                    ambiguous_subject 9 | gold_noise 1
Accuracy by relation (top 5 by n)
  occupation   n=31  acc 12.9%  ███
  author       n=24  acc 25.0%  █████
  ...
=================================================================================
```
- Plain-text tables (pandas `to_string()` or stdlib formatting) and unicode bar charts; no extra dependencies.
- Percentages next to every count. Always show `n`.
- Per stage, print what matters most:
  - **generation:** questions done, time per question, mean answer tokens, % finished cleanly, checkpoint status
  - **correctness:** outcome counts strict vs relaxed, changes vs previous run, review-list size, flag counts, accuracy by relation
  - **entropy:** mean ± std of each feature per label; single-feature AUROC (correct vs wrong) for each feature, sorted; correlation matrix of the entropy levels
  - **labeling:** FA/AH/UH/abstain/flagged counts for each co-occurrence cutoff
  - **probes:** AUROC table by layer × position, best layer marked
  - **classifier/eval:** AUROC / AUPRC / ECE per stage and per class; **comparison table against every baseline** with the best value marked; confusion matrix
- Also write `summary.json` (same numbers) and PNG plots to `outputs/<run>/plots/`: feature histograms per label, ROC curves (method vs baselines), probe AUROC vs layer, reliability diagram. Display the plots inline when running in a notebook.

## 8. Build order — one milestone at a time, test before moving on

1. **M1 (done):** data + generation + correctness, 200 PopQA questions.
2. **M1.5:** correctness fixes in §3 (abstain, strict/relaxed, review list, flags) + §7 reporting for generation and correctness. Rerun `--stages correctness` only.
3. **M2:** `entropy/` all features + sanity plots (each feature for correct vs wrong).
4. **M3:** AH/UH labels with the Wikipedia proxy; manual check of ~100 labels; review ambiguous/gold-noise flags.
5. **M4:** probes.
6. **M5:** classifier + eval with all baselines.
7. **M6:** OLMo + infini-gram proxy validation; scale to all three datasets with `s_pop` stratification.

## 9. Open decisions (defaults in place; revisit with real data)

- **Judge for review-list rows at scale:** NLI vs LLM judge — decide after seeing the M1.5 review list.
- **AH/UH co-occurrence cutoff:** configurable; report 2–3 cutoffs plus manual-check agreement.
- **Automatic ambiguity detection:** Wikidata homonym check — decide at M3 whether the heuristic flag is enough.

## 10. Coding conventions

- Python 3.10+, PyTorch + `transformers`. Type hints, small functions, no notebook-only logic.
- Every stage is config-driven and idempotent: reads inputs from disk, writes outputs to disk.
- Fixed seeds everywhere, logged in `meta.json`.
- `.gitignore` covers `outputs/`, `*.npy`, `*.pt`, checkpoints.
