"""Stage `data`: select questions and assign subject-based splits -> data/questions.jsonl."""

from __future__ import annotations

import random
from typing import Any

from ..utils import run_dir, write_json, write_jsonl, write_meta
from .loaders import load_items
from .splits import assign_subject_splits, split_counts


def select_questions(items: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(items):
        return list(items)
    idx = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in idx]


def run(cfg: dict[str, Any]) -> None:
    dcfg, seed = cfg["data"], cfg["seed"]
    out = run_dir(cfg) / "data"

    items = load_items(dcfg)
    selected = select_questions(items, dcfg.get("n_questions"), seed)
    selected = assign_subject_splits(
        selected, dcfg.get("split_fractions", {"train": 0.6, "val": 0.2, "test": 0.2}), seed
    )

    write_jsonl(out / "questions.jsonl", selected)
    summary = {"n_available": len(items), "n_selected": len(selected), "splits": split_counts(selected)}
    write_json(out / "summary.json", summary)
    write_meta(out, cfg, "data", {"summary": summary})
    print(f"[data] {len(selected)} questions -> {out}  splits={summary['splits']}")
