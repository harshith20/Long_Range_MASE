"""Manual overrides, generic-alias cleaning, flags.

Overrides live in the repo at overrides/<dataset>_<model>.csv with columns
    qid, label, gold_noise, ambiguous_subject, note
  label             correct | wrong | abstain    (blank = keep the automatic label)
  gold_noise        1 / 0                        (blank = 0)
  ambiguous_subject 1 / 0                        (blank = keep the heuristic)
The user gives decisions to Claude Code, which writes them here and commits. Nobody edits
files inside Kaggle.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from ..data.normalize import normalize_answer
from .matching import LABELS

OVERRIDE_COLUMNS = ["qid", "label", "gold_noise", "ambiguous_subject", "note"]
_TRUE, _FALSE = {"1", "true", "yes", "y"}, {"0", "false", "no", "n"}


# ----------------------------------------------------------------------------- aliases

def load_generic_aliases(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return {a for a in (normalize_answer(x) for x in d.get("generic", []) or []) if a}


def clean_aliases(aliases: list[str], generic: set[str]) -> tuple[list[str], list[str]]:
    """Drop single generic words ("church", "film", ...) from the gold aliases.

    PopQA lists "Church" as an alias of "Catholic Church", so any prediction containing
    "church" would match. If every alias is generic, nothing is dropped (and nothing is
    returned as dropped), so the question keeps a gold answer.
    Returns (kept, dropped).
    """
    kept, dropped = [], []
    for a in aliases:
        (dropped if normalize_answer(a) in generic else kept).append(a)
    if not kept:
        return list(aliases), []
    return kept, dropped


# ----------------------------------------------------------------------------- relation groups

def load_relation_groups(path: Path | None) -> tuple[dict[str, str], set[str]]:
    """relation -> 'open' | 'closed', plus the set of borderline relations (§9)."""
    if path is None or not path.exists():
        return {}, set()
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    groups: dict[str, str] = {}
    for g in ("closed", "open"):
        for rel in d.get(g, []) or []:
            key = str(rel).strip().lower()
            if key in groups:
                raise ValueError(f"relation {rel!r} is in both groups in {path.name}")
            groups[key] = g
    borderline = {str(r).strip().lower() for r in d.get("borderline", []) or []}
    return groups, borderline


# ----------------------------------------------------------------------------- flags

def ambiguous_subject_heuristic(subject: str | None, ccfg: dict[str, Any]) -> bool:
    """First-pass flag: single-word subject titles (Emma, Messiah, Boston, Avatar, ...)
    or subjects listed in `ambiguous_subjects`. A column only: it excludes nothing by itself
    (rule 9). The real check runs in M3 on AH-labelled rows."""
    s = normalize_answer(subject or "")
    if not s:
        return False
    extra = {normalize_answer(x) for x in ccfg.get("ambiguous_subjects", []) or []}
    if s in extra:
        return True
    return bool(ccfg.get("ambiguous_single_word", True)) and len(s.split()) == 1


# ----------------------------------------------------------------------------- overrides

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
    """qid -> {label?, gold_noise?, ambiguous_subject?, note?}. Missing file -> {}."""
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            qid = (row.get("qid") or "").strip()
            if not qid:
                continue
            where = f"{path.name}:{i}"
            if qid in out:
                raise ValueError(f"{where}: duplicate qid {qid!r}")
            o: dict[str, Any] = {}
            label = (row.get("label") or "").strip().lower()
            if label:
                if label not in LABELS:
                    raise ValueError(f"{where}: label must be one of {LABELS}, got {label!r}")
                o["label"] = label
            for key in ("gold_noise", "ambiguous_subject"):
                b = _parse_bool(row.get(key), where)
                if b is not None:
                    o[key] = b
            if (row.get("note") or "").strip():
                o["note"] = row["note"].strip()
            out[qid] = o
    return out


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
