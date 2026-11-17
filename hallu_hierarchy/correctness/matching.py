"""Strict / relaxed alias matching and refusal (abstain) detection.

Never judge correctness by embedding similarity: "Ulm" and "Munich" are close in embedding
space. Only normalized whole-word span matching is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..data.normalize import normalize_answer

LABELS = ("correct", "wrong", "abstain")
EMPTY_PATTERN = "<empty>"


def span_in(needle: str, hay: str) -> bool:
    """True if `needle` occurs in `hay` as a whole-word span (both already normalized)."""
    return bool(needle) and bool(hay) and f" {needle} " in f" {hay} "


def match(pred: str, aliases: list[str], min_words: int = 2, min_chars: int = 6) -> dict[str, Any]:
    """Compare a prediction with all gold aliases after normalization.

    match_strict:  a normalized alias appears in the prediction as a whole-word span.
    match_relaxed: strict, or the prediction is a whole-word span of an alias and is at least
                   `min_words` words or `min_chars` characters long ("Drama" does not match
                   "legal drama"; "Munich" matches "Munich Bavaria").
    Relaxed never changes the label; it only puts the row on the review list.
    """
    p = normalize_answer(pred)
    norm_aliases = [a for a in dict.fromkeys(normalize_answer(x) for x in aliases) if a]
    strict = next((a for a in norm_aliases if span_in(a, p)), None)
    long_enough = len(p.split()) >= min_words or len(p) >= min_chars
    sub = next((a for a in norm_aliases if span_in(p, a)), None) if long_enough else None
    relaxed = strict or sub
    return {
        "pred_norm": p,
        "aliases_norm": norm_aliases,
        "match_strict": strict is not None,
        "match_relaxed": relaxed is not None,
        "matched_alias_strict": strict,
        "matched_alias_relaxed": relaxed,
    }


@dataclass(frozen=True)
class RefusalPatterns:
    """Refusal patterns, normalized the same way as answers.

    exact:    the whole normalized answer equals the pattern ("Unknown." -> "unknown")
    contains: the pattern appears as a whole-word span ("X is not a person" -> "is not")
    An empty answer counts as abstain with pattern "<empty>".
    """

    exact: tuple[str, ...]
    contains: tuple[str, ...]

    @classmethod
    def from_lists(cls, exact: list[str], contains: list[str]) -> "RefusalPatterns":
        def _norm(xs: list[str]) -> tuple[str, ...]:
            return tuple(p for p in dict.fromkeys(normalize_answer(x) for x in xs or []) if p)

        return cls(exact=_norm(exact), contains=_norm(contains))

    @classmethod
    def load(cls, path: str | Path) -> "RefusalPatterns":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls.from_lists(d.get("exact", []), d.get("contains", []))

    def find(self, pred_norm: str) -> str | None:
        """The matching pattern, or None if the (normalized) answer is not a refusal."""
        if not pred_norm:
            return EMPTY_PATTERN
        if pred_norm in self.exact:
            return pred_norm
        return next((p for p in self.contains if span_in(p, pred_norm)), None)


def auto_label(m: dict[str, Any], refusal: RefusalPatterns) -> tuple[str, str | None]:
    """(label, refusal pattern). Abstain first, then strict match decides correct/wrong."""
    pattern = refusal.find(m["pred_norm"])
    if pattern is not None:
        return "abstain", pattern
    return ("correct" if m["match_strict"] else "wrong"), None


def relaxed_label(m: dict[str, Any], label_auto: str) -> str:
    """What the label would be under relaxed matching (reporting only, never used as label)."""
    if label_auto == "abstain":
        return "abstain"
    return "correct" if m["match_relaxed"] else "wrong"
