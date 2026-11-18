# Experiment log

One folder per experiment (CLAUDE.md §13). Raw reports: `../reports/`.

| ID | Date | Question | Outcome |
|---|---|---|---|
| [E01](E01_m1_dev200/) | 2026-11-16 | Does the pipeline run end to end on Kaggle? | Yes; 17% accuracy; matching and refusal problems found |
| [E02](E02_m1-5_correctness_dev200/) | 2026-11-17 | Do the M1.5 correctness fixes behave on the 200? | 18.0% correct, 13 refusals separated; ambiguity heuristic flags ~50% |
| [E03](E03_full-popqa_generation/) | 2026-11-17 | Generate all 14,267 PopQA questions | Done in ~2.9 h; 13.6% strict accuracy; relation confounder visible |
| [E04](E04_m1-6_correctness/) | 2026-11-18 | M1.6 labels (head words, `partial`) on all PopQA | PASS; 14.3% correct, 1.2% partial; closed 35% vs open 5% |
