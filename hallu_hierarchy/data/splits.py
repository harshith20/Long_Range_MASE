"""Subject-based splits: one subject never appears in two splits."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any


def assign_subject_splits(
    items: list[dict[str, Any]],
    fractions: dict[str, float],
    seed: int,
    key: str = "subject_id",
) -> list[dict[str, Any]]:
    """Return copies of `items` with a `split` field, grouped by `key`.

    Subjects are shuffled with `seed` and assigned to splits until each split's share of
    *subjects* reaches its fraction. Question counts per split are therefore approximate.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    subjects = sorted({str(it[key]) for it in items})
    random.Random(seed).shuffle(subjects)

    names = list(fractions)
    bounds, acc = [], 0.0
    for name in names:
        acc += fractions[name]
        bounds.append(round(acc * len(subjects)))
    bounds[-1] = len(subjects)

    subj2split: dict[str, str] = {}
    start = 0
    for name, end in zip(names, bounds):
        for s in subjects[start:end]:
            subj2split[s] = name
        start = end

    out = [{**it, "split": subj2split[str(it[key])]} for it in items]
    check_no_leakage(out, key)
    return out


def check_no_leakage(items: list[dict[str, Any]], key: str = "subject_id") -> None:
    seen: dict[str, str] = {}
    for it in items:
        s, sp = str(it[key]), it["split"]
        if seen.setdefault(s, sp) != sp:
            raise AssertionError(f"subject {s!r} appears in splits {seen[s]!r} and {sp!r}")


def split_counts(items: list[dict[str, Any]], key: str = "subject_id") -> dict[str, dict[str, int]]:
    q = Counter(it["split"] for it in items)
    s = Counter(split for _, split in {(str(it[key]), it["split"]) for it in items})
    return {sp: {"questions": q[sp], "subjects": s[sp]} for sp in sorted(q)}


def assign_subject_folds(
    items: list[dict[str, Any]], k: int, seed: int, key: str = "subject_id"
) -> dict[str, int]:
    """subject -> fold in 0..k-1 for subject-grouped k-fold cross-validation (CLAUDE.md §3 Metrics).

    Subjects are shuffled with `seed` and dealt round-robin, so folds differ by at most one
    subject. Every question of a subject is in the same fold (rule 2), and every question is
    tested exactly once across the k folds.
    """
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    subjects = sorted({str(it[key]) for it in items})
    random.Random(seed).shuffle(subjects)
    return {s: i % k for i, s in enumerate(subjects)}


def check_folds(items: list[dict[str, Any]], key: str = "subject_id", fold_key: str = "fold") -> None:
    seen: dict[str, int] = {}
    for it in items:
        s, f = str(it[key]), it[fold_key]
        if seen.setdefault(s, f) != f:
            raise AssertionError(f"subject {s!r} appears in folds {seen[s]} and {f}")
