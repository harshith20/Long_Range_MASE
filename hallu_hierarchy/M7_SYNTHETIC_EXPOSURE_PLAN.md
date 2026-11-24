# M7 — Controlled exposure study: when does a fact become FA, AH or UH, and can the model tell?

Status: E20 (data generator) in progress. Written 2026-11-24; decisions D1–D9 added 2026-11-24 (§1.0). Follows E01–E19 (PopQA × Qwen2.5-3B, see HYPOTHESES.md).
This file is the source of truth for the M7 work. Keep it updated as decisions change; log each run as an
experiment folder per CLAUDE.md §13 (E20, E21, …).

---

## 0. Why this milestone exists

E17–E19 showed, on real data (PopQA × Qwen2.5-3B):
- Uncertainty (entropy, SEP) separates UH from {FA, AH}; AH is confident like FA.
- The correctness probe's FA-vs-AH signal is mostly answer identity: 0.82 overall, **0.545 when FA and AH give
  the identical answer**, 0.59 closed → open transfer.
- AH facts are less salient (gold mentioned 5 vs 11 times on the subject page); 38% of AH show overshadowing;
  12% are role swaps.

On real data we cannot control how often a fact was seen, and labels are a Wikipedia proxy. M7 trains small
models **from scratch** on synthetic data where we **set** every fact's exposure count, so labels are exact
and exposure is a controlled variable.

### Research questions
- **RQ1 (threshold):** How many exposures does a fact need before it is recalled (FA)?
- **RQ2 (self-knowledge):** Do internal states encode how often a fact was seen? Does that separate AH from FA?
- **RQ3 (transitions):** At what exposure count, and at what competitor ratio, does UH turn into AH, and AH into FA?
- **RQ4 (mechanism):** Is AH produced by subject-driven extraction that ignores the relation?

### Hypotheses (write predictions BEFORE running; add to HYPOTHESES.md as H10–H15)
- **H10** Label mix moves UH → AH → FA as exposures rise; AH peaks at middle exposure counts.
- **H11** AH share rises with the competitor ratio (work-city count ÷ birth-city count), log-linearly
  (cf. knowledge overshadowing, Zhang et al. 2025).
- **H12** AH needs a familiar subject: with low subject familiarity, wrong answers are mostly UH (matches H4).
- **H13** Entropy/SEP separates UH from AH at every exposure count, but not AH from FA.
- **H14** A probe can decode log(exposure count) of the asked fact from hidden states (RQ2). Whether this
  separates AH from FA is the key open result.
- **H15** AH answers depend less on the relation: blocking last-token attention to relation tokens changes
  AH outputs less than FA outputs; AH answers often stay the same when the relation is swapped.

---

## 1. Data generator (Phase 0)

Folder: `hallu_hierarchy/m7_synthetic/data/` (D1). Pure Python, CPU only, seeded, deterministic. Output: training token stream
shards + metadata tables. Must print a token budget summary before writing anything.

### 1.1 People and attributes
- Unique invented full names (first + middle + last from separate lists; check no duplicates).
- Attributes per person: `birth_date`, `birth_city`, `university`, `major`, `employer`, `work_city`.
- **birth_city and work_city are drawn from the SAME city list** (about 200 cities) so they can be confused.
  This is what makes AH possible.
- Universities and employers each have a fixed city (e.g. "Northgate University" in a given city), so a
  university or employer city is a third possible AH source.
- All values sampled independently and uniformly (no fact derivable from another). Enforce
  birth_city, work_city, university city and employer city all different (D4).
- City, university, employer and person names are **invented** (syllable-built), so a pretrained model in B1
  has no prior knowledge of them. Majors are real subject names.

### 1.2 Experimental factors (assigned per person)
| factor | levels | what it controls |
|---|---|---|
| `exp_birth` | 1, 3, 10, 30, 100, 300, 1000 | exposures of the target fact (birth city) |
| `ratio` | 0 (work city never mentioned), 1, 10, 30 | competitor strength: work-city exposures = exp_birth × ratio (cap 3000) |
| `familiarity` | low (other attributes 3× each), high (100× each) | how well-known the subject is. "Other attributes" = birth_date, university, major, employer (not work_city, which `ratio` sets) |
| `split` | qa_train, qa_test (50/50) | whether QA pairs about this person appear in training |

Fully crossed: 7 × 4 × 2 = 56 cells (× split). The cap means some cells do not reach their nominal ratio
(exp 300 × 30 → 10; exp 1000 × 10 and × 30 → 3). Each person stores the **effective ratio**; analyses use it
as a continuous variable and flag capped cells (D3). Aim for ≥ 150 people per cell in qa_test → ~17k people.
If the budget is too large, drop the 1000 level or reduce people in high-count cells; report what was cut.

### 1.3 How an exposure is written (same fact, different wording, different context)
- **One exposure = one sentence stating the fact**, in any wording and any context. Count per
  (person, attribute).
- 30–50 templates per attribute; draw a fresh template for every exposure. Example templates for birth city:
  "{name} was born in {city}.", "{city} is the hometown of {name}.", "{name}'s birthplace is {city}."
- Always use the full name (no pronouns).
- Wrap exposures in varied **context types**, chosen at random per document:
  1. biography paragraph (several attributes, shuffled order)
  2. standalone fact sentence
  3. short news-style snippet with filler sentences that contain no facts about this person
  4. list or table row ("Name | Born | Works | …")
  5. mention inside **another person's** document ("{name}'s colleague {name2} was born in {city2}") — counts
     as an exposure for {name2} only, and creates a cross-person co-occurrence to log
- Every sentence that states a fact counts toward that fact's count, whatever the context.
- **Name-mention balancing (D2):** fact-free sentences with the full name ("{name} attended a public lecture.")
  are added so that, within an (exp_birth, familiarity) cell, every person's name appears about as often as in
  the ratio-30 cell. This stops the competitor ratio from also changing how familiar the subject is. Actual
  name-mention counts are stored per person.
- **Co-occurrence partners capped (D4):** each person has at most 3 partners; at most 5 of a person's exposures
  are written in a partner's document. The linked-city set per person and the chance-level AH rate
  (linked cities ÷ 199) are stored.
- Generate all templates by hand or with an LLM once, then freeze them in a file (`templates.json`). Keep a
  held-out set of 5 templates per attribute that never appear in training, for paraphrase tests.

### 1.4 Spreading exposures through training
- Shuffle all documents globally so each fact's exposures are spread uniformly at random across training,
  not clustered.
- Write an **exposure log**: for every (person, attribute), the training step (or token offset) of each
  exposure. This gives, at any checkpoint, the exact "exposures so far" for every fact — a continuous
  exposure axis for free.
- **One epoch, fixed order (D5).** Exposure counts equal sentence counts only if the corpus is read once.
  The training loader reads the token shards in their stored order without reshuffling; the log stores each
  exposure's document token offset, and step = offset ÷ (batch size × context length).
- Shuffling is done with random buckets on disk (uniform permutation without holding the corpus in memory).

### 1.5 QA data (so facts can be retrieved by question)
- For qa_train people only: QA pairs for all attributes ("Q: Where was {name} born? A: {city}"), several
  question templates, mixed into the same pretraining stream (mixed training, as in Allen-Zhu Part 3.1 —
  verify the exact recipe in the paper). **D6:** for now a parameter, default 3 question templates per
  attribute per qa_train person; no work-city questions when ratio = 0 (they would expose the work city).
  Verify the paper's ratio before E21.
- qa_test people get **no** QA pairs. All evaluation uses qa_test people, so a correct answer means real recall
  from biography-style text, not a memorized QA pair.
- QA pairs do not count as exposures in the analysis, but log them.

### 1.6 Tokenizer
- Default: train a small BPE (vocab 8,192) on the generated corpus, so embeddings don't dominate small
  models. Names should split into several tokens (realistic). Record the decision; the same tokenizer is used
  for every model.

### 1.7 Metadata tables (parquet, D8)
- `people.parquet`: person_id, name, all attribute values, factors, split.
- `exposure_log.parquet`: person_id, attribute, exposure_index, step/offset, context_type, template_id.
- `cooccurrence.parquet`: person pairs mentioned together and in which documents.
- `budget.json`: total tokens, tokens per factor level, documents per context type.
- Also: `tokenizer.json`, `tokens_*.bin` (uint16, documents joined with `<|endoftext|>`), `corpus.jsonl`
  (text in training order), `templates.json` copy, `config.yaml` copy.

### 1.0 Decisions taken before E20 (user: "go with recommendations", 2026-11-24)
| id | decision | reason |
|---|---|---|
| D1 | Plan stays at `hallu_hierarchy/M7_SYNTHETIC_EXPOSURE_PLAN.md`; code in `hallu_hierarchy/m7_synthetic/` | CLAUDE.md: work only inside `hallu_hierarchy/` |
| D2 | Fact-free name mentions balance subject familiarity across ratios | otherwise ratio 30 = thousands of extra name mentions, H11 and H12 confounded |
| D3 | Keep the 3,000 cap; store effective ratio; flag capped cells | cap breaks the nominal crossing |
| D4 | All four cities of a person differ; ≤ 3 partners; ≤ 5 cross-document exposures; store chance-level AH | keeps chance AH low and measurable |
| D5 | One epoch, no reshuffle in the loader; log token offsets | exposure = sentence count only for one pass |
| D6 | QA mixing is a parameter (default 3 per attribute); verify Allen-Zhu recipe before E21 | cannot verify the paper now |
| D7 | Templates written by hand, frozen in `templates.json` (statements 35 + 5 held out per attribute; questions 6 + 4 held out) | — |
| D8 | New dependencies: pandas, pyarrow, tokenizers | parquet + BPE |
| D9 | Dry run = 1% of people per cell (2 per cell × split = 224 people, every cell present), full per-person exposure counts | every cell must exist |
| — | ratio 0: work city never stated, so no work-city questions for these people in H15 | intended |

---

## 2. Models and training (Phase 1)

Folder: `m7_synthetic/train/`. Plain PyTorch or HF `transformers` from config (random init, not pretrained).

| ID | architecture | size | where | purpose |
|---|---|---|---|---|
| S25 | GPT-2 style (learned or rotary pos, GELU MLP) | ~25M non-embedding | Kaggle | pilot + fast sweeps |
| S124 | GPT-2 style | ~124M | Kaggle | main run |
| Q124 | Qwen2/Llama style (RoPE, RMSNorm, SwiGLU) | ~124M | Kaggle | architecture check: same size, modern design |
| S350 | GPT-2 style | ~350M | only with RTX 4090 | size trend (optional) |

- 3 seeds per main configuration (S124, Q124). 1 seed for S25 sweeps is fine.
- Context length 512, documents packed.
- AdamW, cosine schedule, warmup 1%, weight decay 0.1. Starting LR: 1e-3 (25M), 6e-4 (124M). Tune only on S25.
- Precision: fp16 with loss scaling on T4 (no bf16 on T4); bf16 on the 4090.
- Attention: PyTorch SDPA (FlashAttention-2 does not support T4).
- Checkpoints: about 12, log-spaced over training, plus final. Save optimizer state at session end so a run can
  resume across Kaggle's 12-hour limit.
- Log train loss and, every checkpoint, a quick QA-test accuracy on 500 people.

### Compute (estimate — replace with measured tokens/s from the pilot)
FLOPs ≈ 6 × params × tokens. At ~200–300M tokens: S25 ≈ 1 h on a T4; S124 ≈ 5–6 h on a T4 (≈ 50 min on a
4090); S350 ≈ 15+ h on a T4 (4090 only, ≈ 2.5 h). Kaggle quota is ~30 GPU-h/week.

---

## 3. Sanity checks (Phase 2) — must pass before any analysis

1. qa_test accuracy rises with `exp_birth` and is near chance (1/200) at 1 exposure.
2. At the highest exposure, high familiarity, ratio 0: qa_test birth-city accuracy is high (target > 80%).
   If not: more templates, more context variety, check the QA mixing.
3. Wrong answers in the competitor cells actually include the work city (AH exists).
4. Paraphrase check: held-out question templates give similar accuracy.
If a check fails, fix the data and regenerate; do not tune the analysis around it.

---

## 4. Labels (exact, no proxy)

For each qa_test question about birth city, from the greedy answer:
- **FA**: answer = true birth city.
- **AH**: answer = a city linked to this person: work city, university city, employer city, or the city of a
  person they co-occur with. Store which link (`ah_source`).
- **UH**: any other city.
- **other**: not a city / malformed (report the rate, exclude).
Also ask the other relations (work city, university) for the same people; needed for H15.

---

## 5. Measurements per checkpoint (Phase 3)

Folder: `m7_synthetic/analysis/`. Run on all qa_test people at every saved checkpoint unless marked (subset).
Store everything per (person, checkpoint) so any later analysis is a table join.

**Axis 1 — uncertainty**
- Greedy token entropy (first answer token, max over answer tokens).
- 10 samples at T=1 → semantic entropy by exact-match grouping (no NLI needed).
- SEP probe (predicts semantic entropy from hidden states), as in E16.

**Axis 2 — correctness probe**
- Linear probe P(wrong) at the first answer token and the last prompt token, all layers; as in E16.

**Axis 3 — is the fact stored? (subject position)**
- Linear probe on the last subject token, each layer, decoding the true birth city (200-way). Does it decode
  the truth even when the output is wrong?

**Axis 4 — competition among candidates (logits)**
- Rank of the true city, probability of the work city, top-1 vs top-2 margin.

**Axis 5 — relation sensitivity (H15)**
- Ask birth city, work city, university city for the same person. Flag when different relations give the same
  answer. Also run the held-out paraphrase templates.

**Axis 6 — mechanism (subset: ~500 people per label)**
- Attention knockout (Geva et al. 2023): block the last position from attending to (a) relation tokens,
  (b) subject tokens, over sliding windows of layers; record the change in probability of the model's own
  answer. Compare FA vs AH vs UH.
- Direct logit attribution by source token (Chughtai et al. 2024): split each head's contribution to the
  answer into subject-token vs relation-token parts.

**Axis 7 — does the model encode exposure count? (RQ2, H14)**
- Ridge regression probe: hidden state → log(exposures so far of the asked fact), at the last subject token
  and last prompt token, each layer. Cross-validate grouped by person. Then test: does the predicted exposure
  separate AH from FA within the same exposure bucket?

Probe rules (all axes): cross-validation grouped by person; train probes per checkpoint; report AUROC within
exposure bucket as well as pooled (same lesson as H5: pooled numbers can be inflated by the grouping variable).

---

## 6. Analyses and figures (Phase 4)

1. **Transition curves (RQ1, RQ3):** share of FA/AH/UH vs log(exposures), one panel per ratio × familiarity.
   Also vs "exposures so far" across checkpoints (from the exposure log).
2. **Threshold:** exposures at which FA reaches 50% and 90%, per ratio and familiarity, per model size and
   architecture.
3. **Two-axis map:** median Axis 1 vs Axis 2 per label per exposure bucket (the FA/AH/UH picture from E18,
   now as a function of exposure).
4. **Detectability curves:** AUROC FA vs AH, AH vs UH, FA vs UH, and 3-class macro, per exposure bucket, for
   each axis alone and combined.
5. **Exposure probe (RQ2):** predicted vs true log exposure; AH vs FA within bucket.
6. **Mechanism (RQ4):** relation-knockout effect and subject/relation attribution, by label.
7. **Generalization:** every main result for S124 vs Q124 vs S25 (and S350 if run). Report where they differ.

---

## 7. Bridge to real models (Phase 5) — needed for the generalization claim

- **B1 — Inject the synthetic facts into a pretrained model.** Continued training of Qwen2.5-0.5B (full
  fine-tune) and Qwen2.5-3B (4-bit + LoRA) on a smaller version of the synthetic corpus (exposures 1–100;
  1000 is too costly). Same labels and measurements. Caveat: fine-tuning in new facts behaves differently
  from pretraining (Zucchet et al. 2025; Gekhman et al. 2024), so treat this as a check, not a replacement.
- **B2 — Real data with counts (M6).** OLMo-2 on PopQA + infini-gram counts of (subject, gold) and
  (subject, predicted answer). Check that the M7 transition picture appears on real frequencies.
- **B3 — Relation knockout on Qwen2.5-3B × PopQA.** Axis 6 on the existing E17 FA/AH rows (7 relations).
  Cheap; tests H15 on a real model.
- **B4 — Hard-negative probe on PopQA (optional).** Train the correctness probe only on FA/AH pairs with the
  identical answer string (4,089 pairs from E18/E19); report the identical-answer AUROC (now 0.545).

---

## 8. Order of work

| step | exp ID | what | needs |
|---|---|---|---|
| 1 | E20 | Data generator + budget printout + 1% dry-run corpus | CPU (local) |
| 2 | E21 | S25 pilot: measure tokens/s, run sanity checks on a small corpus | Kaggle |
| 3 | E22 | Fix data until sanity checks pass; freeze generator v1 | Kaggle |
| 4 | E23 | S124 × 3 seeds, full corpus, all checkpoints | Kaggle (~1 week quota) |
| 5 | E24 | Measurements Axes 1–5 on E23 checkpoints | Kaggle |
| 6 | E25 | Axes 6–7 (subset) | Kaggle |
| 7 | E26 | Q124 × 3 seeds + same measurements | Kaggle |
| 8 | E27 | B3 relation knockout on Qwen2.5-3B PopQA | Kaggle |
| 9 | E28+ | B1 injection, B2 (M6), S350 if a 4090 is available | Kaggle / 4090 |

## 9. Rules for this milestone
- Write predictions for H10–H15 into HYPOTHESES.md before E23 results exist.
- Never overwrite raw outputs; every run gets its own experiment folder and report.
- Seeds and generator version recorded in every report.
- Report within-cell numbers first, pooled second.
- If a result contradicts a hypothesis, record it plainly; do not re-tune the analysis to rescue it.

## 10. Key references (verify each before citing)
- Allen-Zhu & Li — Physics of LMs Part 3.1 (bioS, knowledge augmentation, mixed QA training) and Part 3.3
  (capacity; 100 vs 1000 exposures).
- Zucchet et al. 2025 — learning dynamics of facts, hallucinations appear with knowledge (COLM).
- Chang et al. 2024 — factual knowledge acquisition in OLMo pretraining (NeurIPS).
- Geva et al. 2023 — subject enrichment, relation propagation, attribute extraction; attention knockout.
- Chughtai et al. 2024 — additive mechanisms: subject heads, relation heads, mixed heads.
- Lv et al. 2024; Merullo et al. 2024 — relation-specific heads; argument → function in MLPs.
- Cheang et al. 2025/2026 — AH/UH taxonomy; AH and FA share internal geometry.
- Zhang et al. 2025 — law of knowledge overshadowing.
- Merullo et al. 2025 — linear representations form at subject–object co-occurrence thresholds.
- Kalai et al. 2025 — singleton rate lower-bounds hallucination.
