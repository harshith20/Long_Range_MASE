# Experiment log

One folder per experiment (CLAUDE.md §13). Raw reports: `../reports/`.

| ID | Date | Question | Outcome |
|---|---|---|---|
| [E01](E01_m1_dev200/) | 2026-11-16 | Does the pipeline run end to end on Kaggle? | Yes; 17% accuracy; matching and refusal problems found |
| [E02](E02_m1-5_correctness_dev200/) | 2026-11-17 | Do the M1.5 correctness fixes behave on the 200? | 18.0% correct, 13 refusals separated; ambiguity heuristic flags ~50% |
| [E03](E03_full-popqa_generation/) | 2026-11-17 | Generate all 14,267 PopQA questions | Done in ~2.9 h; 13.6% strict accuracy; relation confounder visible |
| [E04](E04_m1-6_correctness/) | 2026-11-18 | M1.6 labels (head words, `partial`) on all PopQA | PASS; 14.3% correct, 1.2% partial; closed 35% vs open 5% |
| [E05](E05_m1-6b_correctness/) | 2026-11-18 | Religion head words; can H5 be tested within relation? | WARN (capital of refusals); 14.9% correct; only 5/16 relations usable, all closed-set |
| [E06](E06_open-set_label_audit/) | 2026-11-18 | Are open-set "wrong" labels really wrong? (50 audited) | ~2% matching misses; 2 missed refusals; many ambiguous short titles |
| [E07](E07_m1-6c_folds_refusals/) | 2026-11-18 | Labels with 5-fold CV + "not specified" refusals | WARN (capital of); 13/16 relations now usable; abstain 6.6% |
| [E08](E08_subject_states/) | 2026-11-18 | Subject-token hidden states for H4 | PASS; 14,264/14,267 subjects found; 0.057 s/question |
| [E09](E09_m1-6_labels_regenerated/) | 2026-11-19 | Regenerate final labels (with folds) from the generations | WARN (capital of); identical to E07 |
| [E10](E10_m2_prep_lens-check_dry-run/) | 2026-11-19 | Is hidden-state index 36 final-normed? Does M2 run end to end? (local) | Yes (lens = real output, 99.77% top-1); dry run passes |
| [E11](E11_m2_subset200/) | 2026-11-19 | M2 features on 200 questions (real NLI) | WARN (capital of, fp32 fallback); SE AUROC 0.876; levels redundant; subject attention strong |
| [E12](E12_m2_full/) | 2026-11-19 | M2 features on all 14,267 questions | WARN (capital of); within-rel AUROC tok_max 0.797, SE 0.781 (pooled ~0.90); subject attention ≈0.68 within-rel |
| [E13](E13_m3_dev200/) | 2026-11-21 | M3 dev: does the Wikipedia proxy work? (200 wrong + 50 correct, local) | WARN (AH < 30); gold on page 96.5%; AH 8.5% (page rule), almost all closed-set; `disambig_exists` too blunt |
| [E14](E14_m3_full/) | 2026-11-21 | M3 on all 11,113 wrong answers (page rule, ambiguity check, local) | PASS; gold on page 97%; AH 947 before / 740 (`answer_on_disambig`) / 510 (`disambig_exists`); 7 relations allow within-relation AH vs FA |
