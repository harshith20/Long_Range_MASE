"""Review list, manual overrides and flags.

Workflow:
  1. The stage writes <run>/correctness/review/disagreements.csv (strict != relaxed) and
     wrong.csv (all wrong answers, for spotting gold noise). Both have empty columns
     `decision`, `gold_noise`, `ambiguous_subject`, `note`.
  2. Copy the rows you decide on into the overrides file named in the config
     (default hallu_hierarchy/review/<run_name>_overrides.csv, committed to git) and fill:
       decision          correct | wrong | abstain   (blank = keep the automatic label)
       gold_noise        1 / 0                         (blank = not set)
       ambiguous_subject 1 / 0                         (blank = keep the heuristic)
  3. Rerun `--stages correctness`. Outputs are regenerated; overrides are never written
     into outputs by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..data.normalize import normalize_answer
from .matching import LABELS

REVIEW_COLUMNS = [
    "qid", "split", "relation", "subject", "question", "pred", "aliases",
    "match_strict", "match_relaxed", "matched_alias_relaxed", "label_auto", "label",
    "flag_ambiguous_subject",
    "decision", "gold_noise", "ambiguous_subject", "note",
]
_TRUE, _FALSE = {"1", "true", "yes", "y"}, {"0", "false", "no", "n"}


def ambiguous_subject_heuristic(subject: str | None, ccfg: dict[str, Any]) -> bool:
    """First-pass flag: single-word subject titles (Emma, Messiah, Boston, Avatar, ...)
    or subjects listed in `ambiguous_subjects`. A Wikidata homonym check may replace this (M3)."""
    s = normalize_answer(subject or "")
    if not s:
        return False
    extra = {normalize_answer(x) for x in ccfg.get("ambiguous_subjects", []) or []}
    if s in extra:
        return True
    return bool(ccfg.get("ambiguous_single_word", True)) and len(s.split()) == 1


def _parse_bool(value: str | None, where: str) -> bool | None:
    v = (value or "").strip().lower()
    if not v:
        return None
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{where}: expected 1/0/blank, got {value!r}")


def load_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    """qid -> {decision?, gold_noise?, ambiguous_subject?, note?}. Missing file -> {}."""
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            qid = (row.get("qid") or "").strip()
            if not qid:
                continue
            where = f"{path.name}:{i}"
            o: dict[str, Any] = {}
            decision = (row.get("decision") or "").strip().lower()
            if decision:
                if decision not in LABELS:
                    raise ValueError(f"{where}: decision must be one of {LABELS}, got {decision!r}")
                o["decision"] = decision
            for key in ("gold_noise", "ambiguous_subject"):
                b = _parse_bool(row.get(key), where)
                if b is not None:
                    o[key] = b
            if (row.get("note") or "").strip():
                o["note"] = row["note"].strip()
            if qid in out:
                raise ValueError(f"{where}: duplicate qid {qid!r}")
            if o:
                out[qid] = o
    return out


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in REVIEW_COLUMNS})


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
