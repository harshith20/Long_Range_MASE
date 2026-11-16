"""Stage `correctness`: normalized alias matching of greedy (and sampled) answers.

Raw match results and normalized strings are saved so the judging rule can be changed
later without regenerating. Labels are per model: never reuse them across models.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..data.normalize import normalize_answer
from ..utils import read_jsonl, run_dir, write_json, write_jsonl, write_meta

MATCH_MODES = ("exact", "contains")


def match(pred: str, aliases: list[str]) -> dict[str, Any]:
    """Compare a prediction with all gold aliases after normalization.

    exact:    normalized prediction == a normalized alias
    contains: a normalized alias appears as a whole-word span in the normalized prediction
    """
    p = normalize_answer(pred)
    norm_aliases = [a for a in (normalize_answer(x) for x in aliases) if a]
    exact = next((a for a in norm_aliases if p and p == a), None)
    contains = next((a for a in norm_aliases if p and f" {a} " in f" {p} "), None)
    return {
        "pred_norm": p,
        "aliases_norm": norm_aliases,
        "exact_match": exact is not None,
        "contains_match": contains is not None,
        "matched_alias": exact or contains,
    }


def is_correct(m: dict[str, Any], mode: str) -> bool:
    if mode not in MATCH_MODES:
        raise ValueError(f"match_mode must be one of {MATCH_MODES}, got {mode!r}")
    return bool(m["exact_match"] if mode == "exact" else m["contains_match"])


def label_record(rec: dict[str, Any], mode: str) -> dict[str, Any]:
    m = match(rec["greedy"]["text"], rec["aliases"])
    sample_correct = [is_correct(match(s["text"], rec["aliases"]), mode) for s in rec["samples"]]
    return {
        "qid": rec["qid"],
        "split": rec["split"],
        "pred": rec["greedy"]["text"],
        **m,
        "correct": is_correct(m, mode),
        "sample_correct": sample_correct,
        "sample_accuracy": sum(sample_correct) / len(sample_correct) if sample_correct else None,
    }


def summarize(labels: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[bool]] = defaultdict(list)
    for lab in labels:
        by_split[lab["split"]].append(lab["correct"])
    n = len(labels)
    return {
        "n": n,
        "accuracy": sum(lab["correct"] for lab in labels) / n,
        "exact_match_rate": sum(lab["exact_match"] for lab in labels) / n,
        "contains_match_rate": sum(lab["contains_match"] for lab in labels) / n,
        "empty_predictions": sum(not lab["pred_norm"] for lab in labels),
        "accuracy_by_split": {k: sum(v) / len(v) for k, v in sorted(by_split.items())},
    }


def run(cfg: dict[str, Any]) -> None:
    mode = cfg.get("labeling", {}).get("correctness", {}).get("match_mode", "contains")
    rd = run_dir(cfg)
    out = rd / "correctness"
    records = read_jsonl(rd / "generation" / "generations.jsonl")

    labels = [label_record(r, mode) for r in records]
    write_jsonl(out / "labels.jsonl", labels)

    # Wrong answers in a flat file for the manual false-negative check (open decision §8).
    with open(out / "wrong_for_review.tsv", "w", encoding="utf-8") as f:
        f.write("qid\tquestion\tpred\taliases\n")
        for r, lab in zip(records, labels):
            if not lab["correct"]:
                q = r["question"].replace("\t", " ")
                f.write(f"{r['qid']}\t{q}\t{lab['pred']}\t{' | '.join(r['aliases'])}\n")

    summary = summarize(labels)
    write_json(out / "summary.json", summary)
    write_meta(out, cfg, "correctness", {"match_mode": mode, "summary": summary})
    print(f"[correctness] mode={mode} -> {out}  {summary}")
