# Results so far and what they say for each hypothesis

PopQA (14,267 questions) × Qwen2.5-3B-Instruct · status after E17 + E18 (2026-11-23).
All AUROCs are out-of-fold (5-fold CV grouped by subject). **"Within" = macro average of the per-relation
AUROCs** (only relations with ≥ 30 examples per class). This is the number that counts, because pooled numbers
are inflated by relation type (see H5).

Labels: **FA** = correct answer · **AH** = wrong, but the answer is linked to the subject (it appears on the
subject's Wikipedia page) · **UH** = wrong and unlinked. Final counts: FA 2,128 · AH 740 · UH 10,094
(abstain 939, partial 86, ambiguous 207, unscored 72 left out).

---

## 1. Results tables

### 1a. Data and labels (E03–E09, E13–E14)
| | value |
|---|---|
| accuracy (strict) | 14.9% — closed-set relations 37.1%, open-set 5.1% |
| accuracy by subject popularity | U-shaped: lowest decile 20%, middle deciles 7–11%, top decile 47% |
| AH share of wrong answers | 8.6% — 26–62% for capital / genre / religion, 13–21% for family / birthplace / occupation, 1–3% for creators of works |
| AH by group | closed 415 · open 325 |
| proxy check (gold answer on the subject page) | 97% of wrong rows |
| relations usable within-relation | correct vs wrong: 13 of 16 · FA vs AH and AH vs UH: 7 (genre, capital, capital of, religion, place of birth, occupation, father) |
| manual check of the M3 labels | **not done** (user decision) |

### 1b. Single features, correct vs wrong (E12, M2)
| feature (needs sampling?) | pooled | within |
|---|---|---|
| tok_max — greedy token entropy (no) | ~0.90 | **0.797** |
| sem_entropy — semantic entropy, 10 samples + NLI (yes) | ~0.90 | 0.781 |
| seq_nll — sequence NLL over samples (yes) | ~0.89 | 0.766 |
| attention from the last prompt token to the subject (no) | | ≈0.68 (direction flips vs pooled) |
The entropy levels are almost one dimension: Spearman correlations are high and VIF reaches 24.

### 1c. Linear probes on hidden states (E16, M4) — correct vs wrong
| score | input | probe's own target AUROC | pooled | within |
|---|---|---|---|---|
| **SEP for semantic entropy** | answer token (SLT), layer 28 | 0.952 | 0.910 | **0.827** |
| SEP for seq_nll | last prompt token (TBG), layers 32–36 | 0.935 | 0.889 | 0.766 |
| **direct correctness probe** (baseline) | SLT, layer 24 | 0.955 | 0.955 | **0.893** |
| subject familiarity (s_pop high/low) | subject token, layer 20 | 0.822 | 0.587 | 0.714 |
Familiarity probe, mean P(popular subject): **FA 0.578 · AH 0.579 · UH 0.476**.

### 1d. Classifier (E17, M5) — within-relation AUROC (pooled in brackets)
Feature sets: E = token entropy + SEP probes · L = + logit-lens entropy · A = + attention to the subject ·
F = + familiarity probe · C = + correctness probe.
| task | E | E+L+A+F (GB) | E+L+A+F+C | sampled SE | correctness probe alone | relation only |
|---|---|---|---|---|---|---|
| FA vs hallucination (AH+UH) | 0.833 | **0.860** (0.941) | 0.905 | 0.787 | **0.900** | 0.447 (0.852) |
| AH vs UH (positive UH) | 0.730 | 0.728 (0.895) | 0.745 | 0.683 | 0.653 | 0.434 (0.836) |
| FA vs AH | 0.684 | 0.692 (0.798) | **0.807** | 0.648 | 0.800 | 0.428 (**0.754**) |
- Leave-one-relation-out (main LR): FA vs hallucination 0.840; FA vs AH 0.648.
- Without rows carrying the heuristic ambiguous-subject flag: FA vs hallucination 0.802; FA vs AH 0.619.
- Cost after the greedy pass: probes + classifier ≈ 0.015 ms per question (CPU), vs NLI alone 0.375 s per
  question (T4) plus 10 sampled answers.

### 1e. Three classes FA / AH / UH in one model (E17 + E18) — one-vs-rest AUROC, within-relation
| features | FA | AH | UH | macro within | AH recall |
|---|---|---|---|---|---|
| SEP only (1 score) | 0.826 | 0.570 | 0.827 | 0.741 | 0.24 (LR) |
| correctness probe only (1 score) | 0.898 | 0.623 | 0.855 | 0.792 | 0.39 (LR) |
| **SEP + correctness probe (2 scores)** | 0.896 | 0.679 | 0.867 | **0.814** | 0.56 (LR) / 0.67 (GB) |
| E+L+A+F (GB) | 0.851 | 0.658 | 0.855 | 0.788 | 0.57 |
| E+L+A+F+C (GB) | 0.894 | **0.713** | 0.876 | **0.828** | 0.63 |
The flat 3-class model beats the two-stage one (correct-vs-hallucinated, then AH-vs-UH) on AH: 0.629 vs 0.518
within, without C.

### 1f. Diagnostics: where the correctness probe's FA vs AH signal comes from (E18)
| check | FA vs AH | AH vs UH | FA vs UH |
|---|---|---|---|
| correctness probe at the **last prompt token** (before the answer), best layer | 0.698 | 0.63–0.68 | 0.848 |
| correctness probe at the **answer token** (SLT), layers 20–28 | **0.80** | 0.64–0.66 | 0.91 |
| SEP at the answer token, layers 20–28 | 0.66–0.67 | **0.75** | 0.86 |
| "default answer" score (how often the model gives the same answer in the relation) | 0.58 | 0.44 | 0.53 |
| correctness probe, FA vs AH **with the identical answer string** (101 relation×answer groups, 4,089 pairs) | **0.61** (SLT) / 0.60 (TBG) | | |
Median probe scores: SEP (P high entropy) FA 0.03 · AH 0.15 · UH 0.95; correctness probe (P wrong) FA 0.08 ·
**AH 0.64** · UH 0.97.

### 1g. Four AH mechanism tests (E19)
| test | result (within-relation AUROC) |
|---|---|
| correctness probe trained on closed-set, tested on open-set | correct vs wrong 0.828 (in-group 0.896) · **FA vs AH 0.593** (in-group 0.746) |
| correctness probe trained on open-set, tested on closed-set | correct vs wrong 0.770 (in-group 0.892) · FA vs AH 0.743 (in-group 0.862) |
| probe trained directly on FA vs AH (answer token, layers 20–28) | **0.824** (open 0.796, closed 0.908) |
| … the same probe when FA and AH give the identical answer string | **0.545** |
| gold-answer mentions on the subject's page (median) | FA 11 · AH 5 · UH 5 → as a score: FA vs AH 0.686, FA vs UH 0.731 |
| AH answer mentioned more often than the gold on the subject page | 38% of AH |
| answer is a valid value of the asked relation | FA 93% · AH 47% · UH 36% |
| role swap (answer is only a value of another relation) | AH 12% (county → state, town → capital city, director → producer) · UH 7% |

---

## 2. Each hypothesis: the claim, and what the results say

**H1 — The entropy hierarchy separates UH from FA.**
*Supported.* FA vs UH: SEP probe 0.860 within, sampled SE 0.807, and the full single-pass classifier reaches
0.86 on FA vs hallucination. The hierarchy beats every single entropy level (best single 0.833 → 0.860). The
gain from combining levels is small, because the levels are nearly one dimension (VIF up to 24).

**H1b — Single-pass probes keep most of the performance of full sampling (the edge claim).**
*Supported, and stronger than claimed.* The single-pass SEP probe (0.827) beats sampled semantic entropy (0.781)
outright. The single-pass classifier (0.860) beats it by 0.07. The extra cost after the greedy pass is ≈ 0.015 ms
per question, vs ≥ 0.375 s for NLI alone plus 10 samples. Open item: a timing run on the T4 that splits the
greedy pass from the sampling.

**H2 — Entropy alone fails on AH vs FA (AH is confident and consistent).**
*Supported (not at chance, but much weaker).* Entropy-only FA vs AH is 0.65–0.68 within-relation, vs 0.83–0.86
for FA vs UH. In open-set relations it is close to chance (SEP 0.54). AH answers have low entropy, like correct
answers (median SEP score FA 0.03, AH 0.15, UH 0.95). The model is confident when it gives an AH.

**H3 — Non-confidence internal signals (ℓ*, layer-wise entropy, attention to the subject) partly recover AH vs FA.**
*Not supported for these features.* Adding logit-lens entropy, ℓ*, attention and familiarity leaves FA vs AH at
0.67–0.69 within-relation (entropy only: 0.684). **But the hidden state does contain the signal:** the direct
correctness probe reaches 0.800, and 0.807 with everything. **E19 qualifies this:** a probe trained directly on
FA vs AH reaches 0.824, but only **0.545 when FA and AH give the identical answer**, and the correctness probe's
FA vs AH skill falls to 0.59 when it moves from closed- to open-set relations. Most of the hidden-state FA vs AH
signal is *which answer was given* (an answer/relation prior), not a transferable "this fact is wrong" signal.
Verdict: H3 is negative. AH stays hard even for hidden-state probes once the answer is controlled, which
strengthens H2.

**H4 — Subject familiarity separates UH but not AH (AH = familiar subject + unknown fact).**
*Supported clearly.* The familiarity probe gives the same score to FA and AH subjects (0.578 vs 0.579; FA vs AH
pooled 0.516) and a lower one to UH (0.476). In the classifier, familiarity is the feature that helps AH vs UH
and does nothing for FA vs AH. Note: familiarity predicts correctness better within relations (0.714) than
pooled (0.587), because accuracy is U-shaped in popularity across relations.

**H5 — Detection depends on relation type; pooled AUROC is inflated by it.**
*Supported strongly.* Every pooled number sits 0.06–0.15 above its within-relation value. For FA vs AH, the
pooled AUROC of the full model (0.757) equals the **relation-only** baseline (0.754): pooled AH detection is
detecting question type. Leave-one-relation-out is close to within-relation for FA vs hallucination (0.840) but
low for FA vs AH (0.648). AH is concentrated in closed-set relations (26–62% of wrong answers vs 1–3% for
creators of works), and every AH test rests on only 7 relations.

**H6 — AH answers change more under paraphrasing than FA answers.** *Not tested yet* (optional stage).

**H7 — Many errors on ambiguous subjects are not hallucinations.** *Partly tested (indirectly).* 207 AH rows
had the model's answer on the subject's disambiguation page. A spot check showed the rule also fires when the
answer appears only in the line about the subject itself (e.g. "Dallas … in Texas"), so ≥ 12 of the 207 are
really AH. Removing the heuristic ambiguous-subject rows lowers within-relation AUROC (0.843 → 0.802), so
ambiguous subjects are not what makes detection work. The regeneration test with full Wikipedia titles has not been run.

**H8 — The detector transfers across datasets.** *Not tested yet* (needs TriviaQA / EntityQuestions).

**H9 — Errors come from rare facts (fact frequency).** *First evidence with a Wikipedia proxy (E19); the real
test is M6 (OLMo + infini-gram).* The gold answer is mentioned about half as often on the subject's page for AH
and UH (median 5) as for FA (11), and this "fact salience" alone separates FA vs AH (0.686) as well as entropy
does. In 38% of AH the wrong answer is mentioned more than the gold (overshadowing). Subject popularity alone
(s_pop) is not fact frequency: accuracy is U-shaped in it.

**M7 hypotheses (synthetic exposure study; plan `../M7_SYNTHETIC_EXPOSURE_PLAN.md`).** Predictions written
2026-11-24, before any M7 model was trained (plan §9). All *not tested yet*.

**H10 — Label mix moves UH → AH → FA as exposures rise.** *Open (E22): the switch from unknown to known happens between exp 1 and 10, so the grid is too coarse; AH 14% → 19% of wrong answers from exp 1 to 3.* Prediction: as the birth-city
exposure count rises, wrong answers shift from UH to AH and then to FA; the AH share peaks at middle exposure counts.

**H11 — AH share rises with the competitor ratio.** *Supported, one seed (E22): AH share of wrong answers 5.3 / 9.0 / 18.7 / 27.8% at ratio 0 / 1 / 10 / 30 (chance 3.8%), almost all = the work city.* Prediction: the AH share rises with the
competitor ratio (work-city count ÷ birth-city count), log-linearly (cf. knowledge overshadowing, Zhang et al. 2025).

**H12 — AH needs a familiar subject.** *Supported, one seed (E23): AH share of wrong answers at ratio 30 is 37% high vs 19% low familiarity (ratio 1: 15% vs 5%).* Prediction: with low subject familiarity, wrong answers
are mostly UH (matches H4).

**H13 — Entropy separates UH from AH, not AH from FA, at every exposure count.** *Not supported (E23, exp 1–3): FA vs AH 0.77, AH vs UH 0.63; AH here is low-confidence fallback, not confident error.* Prediction:
entropy / SEP separate UH from AH at every exposure count, but not AH from FA.

**H14 — Hidden states encode how often a fact was seen.** *Not tested yet.* Prediction: a probe can decode
log(exposure count) of the asked fact from hidden states (RQ2). Whether this separates AH from FA is the key open result.

**H15 — AH answers depend less on the relation.** *Not tested yet.* Prediction: blocking last-token attention to
the relation tokens changes AH outputs less than FA outputs; AH answers often stay the same when the relation is swapped.

---

## 3. Why the correctness probe separates FA vs AH but not AH vs UH

**What each task contains.** Yes, your assumption is right. AH vs UH uses **only wrong answers** (FA rows are
removed). FA vs AH uses only correct answers and AH.

**Why the correctness probe can't split AH from UH.** It was trained on *correct vs wrong*, and both AH and UH
were "wrong" in its training data. Nothing in its objective asks it to rank AH against UH. Its 0.65 there is a
side effect: AH looks a little more "correct-like". The axis that does separate AH from UH is **uncertainty**:
SEP at the answer token gives 0.75. UH is an uncertain guess; AH is a confident answer.

**The picture is two axes, not one.**
- Axis 1, *uncertainty* (entropy, SEP): FA low · **AH low** · UH high.
- Axis 2, *"is this answer wrong"* (correctness probe): FA low · **AH middle–high (0.64)** · UH high.
- So **AH is confident but wrong**: low on axis 1, high on axis 2. One axis is never enough. That is why two
  scores (SEP + correctness probe) already give a 3-class macro AUROC of 0.814, close to the full model (0.828).
  This also argues for the **3-class formulation** over two-stage: the two-stage pipeline throws away the
  confident-but-wrong cell at stage 1.

**What the correctness probe reads (E18 diagnostics):**
1. **Mostly after the answer is written.** At the last prompt token (before answering) FA vs AH is only ~0.70,
   about the same as entropy. At the answer token it is 0.80. Half the signal appears once the answer is in the
   residual stream. It looks like a check of *this answer for this subject*, not a "do I know this fact" signal
   computed in advance.
2. **Partly answer identity.** With the identical answer string (e.g. "Rock" correct for one band, AH for another)
   the probe drops from 0.80 to ~0.61. So a good share of its power is "which answers tend to be right in this
   relation". That is a relation/answer prior, and it partly explains why the probe is strongest in closed-set
   relations (0.88 vs 0.74 open).
3. **Not "default answers".** AH answers are not mainly the model's go-to answers for a relation (answer-frequency
   score 0.58 for FA vs AH; AH and FA repeat answers equally often).
4. **A residual consistency signal remains** (0.60–0.61 with the same answer string, also before the answer). Some
   internal "subject–answer fit" signal exists beyond the answer prior. This is the interpretable signal to look for.

5. **E19 settles most of this:** a probe trained *only* on FA vs AH gets 0.82, but 0.545 with identical answers,
   and the FA vs AH skill does not transfer between relation groups (0.59). The "knows it is wrong" signal for AH
   is small. What looked like recovery is mostly the answer prior.
6. **What does distinguish AH (E19):** less salient facts (the gold is mentioned 5 vs 11 times on the subject page),
   overshadowing in 38% of cases, containment / role swaps in 12% (county → state, town → capital city,
   director → producer). AH answers mostly have the right type (47% valid for the relation).

**Working mechanism hypothesis for AH (to test):** the subject is recognised (familiar, H4), and the
subject-enrichment step works. But the *relation-specific* attribute extraction retrieves the **most strongly
associated** entity instead of the asked attribute (Dallas → Texas, The Departed → Scorsese, Alois Rašín →
Prague): a dominant association overshadows the specific fact. The output is confident because one strongly
associated candidate wins (low entropy). Later layers, once the answer is in the stream, partly "notice" the
mismatch, which is what the probe at the answer token reads.

**Experiments this suggests (before or alongside M6):**
- **AH vs FA probe directly on hidden states** (baseline / upper bound only): is FA vs AH linearly separable
  beyond 0.80, and at which layer and position?
- **Relation–answer match feature:** does the answer have the right *type* for the relation (a person who is a
  producer vs a director, a county vs a state)? Many AH look like "right entity type, wrong role".
- **Answer–subject association strength** (co-occurrence of subject and answer vs subject and gold), from the
  Wikipedia cache now and infini-gram in M6. This connects AH directly to H9 / knowledge overshadowing.
- **Attention / attribute-extraction analysis at the answer position:** which tokens the last layers attend to
  for FA vs AH (relation tokens vs subject tokens), following the "attribute extraction" view below.
- **Generalisation check:** train the correctness probe on closed-set relations and test it on open-set
  (and vice versa). If it collapses, the probe is mostly an answer/relation prior.

---

## 4. Papers to read (from memory — verify each citation before using it)
- **Kossen et al. 2024, "Semantic Entropy Probes"** — the SEP method used here; probes on hidden states predict
  semantic entropy, and also accuracy probes as the comparison point.
- **Farquhar et al. 2024 (Nature), "Detecting hallucinations in LLMs using semantic entropy"** — the sampled
  SE baseline; it is built for confabulations (≈ UH), not consistent errors (≈ AH).
- **Orgad et al. 2024/2025, "LLMs Know More Than They Show"** — truthfulness signals concentrate in the *exact
  answer tokens* (matches our SLT > TBG), error-type taxonomy, probes generalise poorly across tasks. Closest to section 3.
- **Simhi et al. 2024, "Distinguishing Ignorance from Error in LLM Hallucinations"** — hallucinating *despite
  knowing* vs not knowing; a close cousin of AH vs UH.
- **Geva et al. 2023, "Dissecting Recall of Factual Associations in Auto-Regressive LMs"** — subject enrichment
  in early MLPs, then relation-driven attribute extraction by attention at the last token. A frame for "AH = extraction picked the wrong attribute".
- **Meng et al. 2022, "Locating and Editing Factual Associations in GPT" (ROME)** — causal tracing of where facts are stored.
- **Ferrando et al. 2024/2025, "Do I Know This Entity? Knowledge Awareness and Hallucinations in LMs"** — SAE
  latents for known vs unknown entities. Matches H4: entity recognition says "known" for AH.
- **Gottesman & Geva 2024, "Estimating Knowledge in LLMs Without Generating a Single Token" (KEEN)** — probes on
  subject representations predict how much the model knows. Same idea as the familiarity probe.
- **Kadavath et al. 2022, "Language Models (Mostly) Know What They Know"** — P(IK) before answering. Our TBG result
  (~0.70 on FA vs AH) says this is weak for AH.
- **Azaria & Mitchell 2023 (SAPLMA); Marks & Tegmark 2023, "The Geometry of Truth"** — truth directions in
  hidden states; background for why a linear correctness probe works.
- **Kang & Choi 2023, "Impact of Co-occurrence on Factual Knowledge of LLMs"** — models prefer frequently
  co-occurring answers. A direct candidate mechanism for AH, and the bridge to H9.
- **Zhang et al. 2024/2025, "Knowledge Overshadowing"** — dominant knowledge overshadows less popular knowledge
  and causes hallucination. Same mechanism hypothesis as section 3.
- **Yuksekgonul et al. 2024, "Attention Satisfies" (SAT Probe)** — attention to constraint tokens predicts factual
  errors. Relevant to why our attention feature did not help AH, and to how to define it better.
- **Mallen et al. 2023 (PopQA), Kandpal et al. 2023 (long-tail knowledge)** — popularity / frequency and recall.
