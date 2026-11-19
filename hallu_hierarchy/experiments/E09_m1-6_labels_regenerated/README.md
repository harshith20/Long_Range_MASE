# E09 — Final M1.6 labels regenerated (E07 rerun)

2026-11-19 · commit 8992b47 · stage correctness · GPU off · kernel v8 · ~2 min · **STATUS ⚠️ WARN** (expected)
Report: [`reports/2026-11-19_qwen2.5-3b_popqa_full_correctness_v8.md`](../../reports/2026-11-19_qwen2.5-3b_popqa_full_correctness_v8.md) · reruns [E07](../E07_m1-6c_folds_refusals/)

## Why
The final labels with folds existed only in kernel v6's output, in no dataset. Rerun correctness on the
generations (dataset `hallu-hierarchy-m-1-5` v3) so the labels can be saved as a dataset for M2/M3.

## Setup
No correctness code change since E07 (6de0783). Input: dataset v3 (generations + pre-M1.6 labels as "previous").

## Results
Identical to E07: every number in KEY NUMBERS and FOR CLAUDE matches, incl. fold sizes
(2,846 / 2,889 / 2,824 / 2,851 / 2,857). correct 2,128 (14.9%) | wrong 11,114 | abstain 939 (6.6%) | partial 86.
Only differences: commit and run time (14.6 s vs 10.8 s).

## Insights
- Correctness and fold assignment are deterministic across reruns — labels are reproducible from the generations.
- Same `capital of` WARN (34% abstain), as in E05/E07.
- "Changed vs previous run" compares to the pre-M1.6 labels in the input dataset (as in E07), not to E07.

## Next steps
1. Save kernel v8's output as a dataset (labels with folds) so later runs attach it.
2. Add the subject-states and labels datasets to `kernel-metadata.json`; add `subject_states` to the folders
   `utils.discover_kaggle_inputs` recognises.
3. M2 on a 200-question subset.
