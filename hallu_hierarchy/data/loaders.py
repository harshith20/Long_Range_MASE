"""Dataset loaders. Each returns a list of question dicts with a common schema:

    qid, question, subject, subject_id, aliases (list[str]), meta (dataset-specific)
"""

from __future__ import annotations

import json
from typing import Any, Callable


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    return [str(v) for v in value if str(v).strip()]


def load_popqa(dcfg: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dcfg.get("hf_id", "akariasai/PopQA"), split=dcfg.get("hf_split", "test"))
    required = {"id", "question", "subj", "subj_id", "possible_answers"}
    missing = required - set(ds.column_names)
    if missing:
        raise KeyError(f"PopQA is missing columns {missing}; got {ds.column_names}")

    items = []
    for row in ds:
        aliases = _parse_list(row["possible_answers"])
        if not aliases:
            continue
        items.append(
            {
                "qid": f"popqa-{row['id']}",
                "question": row["question"],
                "subject": row["subj"],
                "subject_id": str(row["subj_id"]),
                "aliases": aliases,
                "meta": {
                    k: row.get(k)
                    for k in ("prop", "obj", "s_pop", "o_pop", "s_wiki_title", "o_wiki_title")
                },
            }
        )
    return items


def _not_yet(name: str) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    def _f(dcfg: dict[str, Any]) -> list[dict[str, Any]]:
        # TriviaQA / NQ-Open have no subject field; subject extraction is needed for the
        # subject-based split and is scheduled for M6.
        raise NotImplementedError(f"{name} loader is scheduled for milestone M6")

    return _f


LOADERS: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
    "popqa": load_popqa,
    "triviaqa": _not_yet("triviaqa"),
    "nq_open": _not_yet("nq_open"),
}


def load_items(dcfg: dict[str, Any]) -> list[dict[str, Any]]:
    name = dcfg["dataset"]
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](dcfg)
