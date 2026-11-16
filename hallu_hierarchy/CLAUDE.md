# CLAUDE.md — hallu_hierarchy

This folder is a self-contained research project inside a larger repo. Work only inside
`hallu_hierarchy/` unless told otherwise. Do not modify code outside this folder.

## 1. What this project is

Thesis work: detect LLM hallucinations with **semantic entropy probes (SEPs)** and a
**multi-level entropy hierarchy**, using the entropy levels as a *joint feature vector* for a
**learned classifier boundary** instead of a hand-set threshold.

Three answer classes, studied separately:

| Label | Meaning |
|---|---|
| **FA** — factual association | Model's answer matches the gold answer |
| **AH** — associated hallucination | Wrong, but the wrong answer is strongly linked to the subject (e.g. "Where was Einstein born?" → "Munich") |
| **UH** — unassociated hallucination | Wrong, and the answer has little or no link to the subject (→ "Lisbon") |

### Hypotheses
- **H1:** The entropy hierarchy separates UH from FA well.
- **H2:** Entropy alone largely fails on AH vs FA (AH is confident and consistent across samples → low entropy).
- **H3:** Adding non-confidence signals (layer-wise dynamics, attention to subject tokens) partly recovers AH vs FA.

A clearly documented negative result on H2 is acceptable and expected. Do not tune things to hide it.

## 2. Non-negotiable rules

1. **Labels never become features.** FA/AH/UH come from gold answers + co-occurrence. Semantic entropy is a feature, never a label.
2. **Split by subject entity, not randomly.** The same subject must never appear in both train and test.
3. **Inference must stay cheap (edge setting).** Sampling + NLI is allowed only to *create training targets*. At inference time, only single-pass signals are allowed: probe outputs, greedy-pass token entropy, layer-wise features, classifier.
4. **Entropy levels are correlated, not independent.** Report the correlations or VIF between them. Include explicit *disagreement* features (e.g. low token entropy but high semantic entropy).
5. **Labels are per model.** The same question can be FA for Qwen and UH for OLMo. Never share labels across models.
6. **Never commit outputs.** Hidden states and generations go to `/kaggle/working/` (or `outputs/` locally, gitignored).

## 3. Pipeline

```
QA data → generation (greedy + N samples, cache hidden states)
        ├→ labeling: correct? → FA ; wrong → AH / UH via co-occurrence
        └→ entropy hierarchy: token, sequence, semantic, layer-wise
              → probes on hidden states (predict each entropy level from 1 pass)
                    → boundary classifier (features → FA / AH / UH)
                          → evaluation vs baselines, per dataset
```

### Generation
- Prompt: short-form QA, answer in a few words. Use the chat template of each model.
- 1 greedy answer (temperature 0) + N = 10 samples (temperature 1.0), max ~32 new tokens.
- Cache hidden states for the **greedy** pass at two positions:
  - **TBG**: last token of the prompt, before generation
  - **SLT**: second-last token of the generated answer
- Cache a subset of layers (e.g. every 4th layer + the last), stored as fp16 `.npy`/`.pt`.
- Also save per-token logits entropy for the greedy answer and log-probs of all samples.
- Also save attention to subject tokens (mean over heads, selected layers) for the greedy pass.

### Labeling
1. **Correctness:** normalize (lowercase, strip punctuation and articles) and match against
   *all* gold aliases. Save the raw match result plus the normalized strings so the judging
   method can be changed later without regenerating.
2. **AH vs UH** (wrong answers only), by subject–answer co-occurrence:
   - OLMo: counts in OLMo's open pretraining corpus via infini-gram (check that the index matches the model's training data version).
   - Qwen: Wikipedia proxy (answer appears on the subject's page / count of pages mentioning both).
   - High co-occurrence → AH, low → UH. The cutoff is a parameter, not a constant (see §8).
3. **Proxy validation:** on OLMo, label the same answers with both the corpus counts and the Wikipedia proxy, then report agreement. This justifies using the Wikipedia proxy for Qwen.

### Entropy hierarchy (features)
- **Token level:** mean / max per-token entropy of the greedy answer.
- **Sequence level:** length-normalized negative log-likelihood across samples.
- **Semantic level:** cluster the N samples by bidirectional entailment (DeBERTa-large-MNLI) and take the entropy over the clusters.
- **Layer level:** logit-lens entropy at each cached layer, plus the layer where the final answer first becomes top-1.
- **Disagreement features:** differences and ratios between levels.

### Probes
- Linear (logistic regression) probes on cached hidden states, one per entropy level, as in the SEP setup (binarize the target entropy at a training-set split point).
- Report probe performance per layer and per position (TBG vs SLT).

### Classifier
- Start with a **two-stage** design: (1) correct vs hallucinated, (2) AH vs UH. This shows *where* it breaks.
- Begin with logistic regression for interpretability, then try gradient boosting.
- Also report a flat 3-class version.

### Baselines (required)
- Semantic entropy (full sampling), SEP alone, token entropy alone
- **Direct correctness probe** (linear probe predicting correct/wrong from hidden states). If the hierarchy does not beat this, say so.
- The previous single-threshold entropy-hierarchy method from earlier thesis work

### Metrics
- AUROC and AUPRC per class (one-vs-rest) and per stage; calibration (ECE); per-dataset tables.

## 4. Models and datasets (Kaggle free tier: one T4, 16 GB)

Verify the exact Hugging Face IDs before use.

| Role | Model |
|---|---|
| Main | `Qwen/Qwen2.5-3B-Instruct` (fp16, ~6 GB) |
| Open-pretraining-data check | OLMo-2 1B Instruct (`allenai/OLMo-2-0425-1B-Instruct`) |
| Semantic clustering | `microsoft/deberta-large-mnli` |

- No 7B models: on a T4 they need 4-bit quantization, which alters the hidden states being probed.

| Dataset | Why |
|---|---|
| PopQA | Popularity score per subject → rare vs famous; expect more UH on rare subjects |
| TriviaQA (rc.nocontext) | Mostly well-known topics; expect more AH |
| NQ-Open | Real search queries, a realistic mix |

Start with ~1,000 questions per dataset. Development runs use 200 PopQA questions.

## 5. Kaggle workflow

- Code lives in this repo. The Kaggle notebook is thin: clone the repo → install → `python -m hallu_hierarchy.run --config configs/<file>.yaml`.
- Secrets (GitHub token for a private repo, HF token) come from Kaggle Secrets. Never hardcode them.
- Sessions can die (~12 h limit, weekly GPU quota). Generation must **checkpoint every 100 questions** and **resume** from the last checkpoint.
- Save finished outputs as a Kaggle Dataset so later stages (probes, classifier) can load them without a GPU rerun.
- Every output folder gets a `meta.json` containing the git commit hash, model ID, dataset, seed, full config and timestamp.

## 6. Layout

```
hallu_hierarchy/
  CLAUDE.md
  configs/        # one yaml per model × dataset
  data/           # dataset loaders, alias handling, subject-based splits
  generation/     # greedy + samples, hidden-state / attention caching, checkpointing
  labeling/       # correctness, co-occurrence, AH/UH assignment
  entropy/        # token, sequence, semantic, layer-wise, disagreement features
  probes/         # SEP training per level / layer / position
  classifier/     # two-stage + flat classifiers
  eval/           # metrics, baselines, plots, per-dataset tables
  run.py          # entry point, dispatches stages by config
  requirements.txt
```

## 7. Build order — do one milestone at a time, and test it before the next

1. **M1:** `data/` + `generation/` + correctness labeling. Run Qwen on 200 PopQA questions on Kaggle. Output: answers, samples, hidden states, correct/wrong. *Do not start M2 until M1 runs end-to-end on Kaggle.*
2. **M2:** `entropy/` (all levels) + quick sanity plots (entropy for correct vs wrong).
3. **M3:** `labeling/` AH/UH with the Wikipedia proxy; manual check of ~100 labels.
4. **M4:** `probes/`.
5. **M5:** `classifier/` + `eval/` with all baselines.
6. **M6:** OLMo + infini-gram proxy validation; scale to all three datasets.

## 8. Open decisions (defaults in place; revisit with real data)

- **Correctness judging:** default is normalized alias matching. If manual inspection shows many false negatives (correct answers phrased differently), add an NLI- or LLM-based judge as a second pass.
- **AH/UH co-occurrence cutoff:** keep it configurable; report results for 2–3 cutoffs plus the manual-check agreement.

## 9. Coding conventions

- Python 3.10+, PyTorch + `transformers`. Type hints, small functions, no notebook-only logic.
- Every stage is config-driven and idempotent: it reads inputs from disk and writes outputs to disk.
- Fixed seeds everywhere and logged in `meta.json`.
- Use `.gitignore` for `outputs/`, `*.npy`, `*.pt`, checkpoints.
