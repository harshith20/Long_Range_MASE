# M7 data generator

Spec: `../../M7_SYNTHETIC_EXPOSURE_PLAN.md` §1 (decisions D1–D9 in §1.0).

```
# token budget of the full design only (writes nothing)
python -m hallu_hierarchy.m7_synthetic.data.generate --budget-only
# 1% dry run (2 people per cell × split)
python -m hallu_hierarchy.m7_synthetic.data.generate --fraction 0.01 --out hallu_hierarchy/outputs/m7_synthetic/<name>
```

| file | content |
|---|---|
| `config_v0.yaml` | factors, context weights, QA mixing, tokenizer, seed |
| `templates.json` | frozen statement / question templates (train + held out), filler, table headers |
| `lexicon.py` | invented names, cities, universities, employers |
| `generate.py` | people → plan → budget → render into random buckets → BPE → corpus in training order |

Outputs: `people.parquet`, `exposure_log.parquet` (one row per exposure and QA pair, with document token offset),
`cooccurrence.parquet`, `documents.parquet`, `tokens_*.bin` (uint16), `corpus.jsonl`, `tokenizer.json`,
`budget.json`, copies of config and templates. An existing non-empty output folder is never overwritten.
The corpus must be read **once, in stored order** (D5).
