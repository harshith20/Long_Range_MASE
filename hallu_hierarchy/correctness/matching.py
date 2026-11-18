"""Strict / relaxed alias matching, head-word stripping and refusal (abstain) detection.

Never judge correctness by embedding similarity: "Ulm" and "Munich" are close in embedding
space. Only normalized whole-word span matching is used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..data.normalize import normalize_answer

LABELS = ("correct", "wrong", "abstain", "partial")
EMPTY_PATTERN = "<empty>"


def span_in(needle: str, hay: str) -> bool:
    """True if `needle` occurs in `hay` as a whole-word span (both already normalized)."""
    return bool(needle) and bool(hay) and f" {needle} " in f" {hay} "


# ----------------------------------------------------------------------------- head words

def load_head_words(path: str | Path | None) -> tuple[str, ...]:
    """Normalized head words, longest first so "tv series" is tried before "series"."""
    if path is None or not Path(path).exists():
        return ()
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    words = {normalize_answer(x) for x in d.get("head_words", []) or []} - {""}
    return tuple(sorted(words, key=lambda w: (-len(w.split()), -len(w), w)))


def strip_head_words(alias_norm: str, head_words: tuple[str, ...]) -> str | None:
    """Remove trailing generic head words: "comedy film" -> "comedy",
    "science fiction film" -> "science fiction". Returns None if nothing was stripped
    or nothing would remain ("film" alone is never reduced to an empty alias)."""
    out, changed = alias_norm, True
    while changed:
        changed = False
        for hw in head_words:
            if out.endswith(" " + hw) and len(out) > len(hw) + 1:
                out = out[: -len(hw) - 1].strip()
                changed = True
                break
    return out if out != alias_norm and out else None


# ----------------------------------------------------------------------------- matching

def match(
    pred: str, aliases: list[str], min_words: int = 2, min_chars: int = 6,
    head_words: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Compare a prediction with all gold aliases after normalization.

    match_strict:  a normalized alias, or its head-word-stripped copy, appears in the
                   prediction as a whole-word span ("Comedy" vs "comedy film" -> strict).
    match_relaxed: the prediction is a whole-word span of an alias (original or stripped) and
                   is at least `min_words` words or `min_chars` characters long
                   ("Drama" does not match "legal drama"; "Singer" matches "singer songwriter").
    Rows that match only under relaxed become `partial`, never `correct`.
    """
    p = normalize_answer(pred)
    norm_aliases = [a for a in dict.fromkeys(normalize_answer(x) for x in aliases) if a]
    stripped = [s for s in dict.fromkeys(strip_head_words(a, head_words) for a in norm_aliases) if s]
    stripped = [s for s in stripped if s not in norm_aliases]

    strict_orig = next((a for a in norm_aliases if span_in(a, p)), None)
    strict_head = None if strict_orig else next((s for s in stripped if span_in(s, p)), None)
    strict = strict_orig or strict_head
    long_enough = len(p.split()) >= min_words or len(p) >= min_chars
    sub = next((a for a in norm_aliases + stripped if span_in(p, a)), None) if long_enough else None
    relaxed = strict or sub
    return {
        "pred_norm": p,
        "aliases_norm": norm_aliases,
        "aliases_stripped": stripped,
        "match_strict": strict is not None,
        "match_relaxed": relaxed is not None,
        "strict_via_head_word": strict_head is not None,
        "matched_alias_strict": strict,
        "matched_alias_relaxed": relaxed,
    }


# ----------------------------------------------------------------------------- refusals

@dataclass(frozen=True)
class Pattern:
    text: str
    except_relations: frozenset[str] = field(default_factory=frozenset)


def _parse_patterns(items: list[Any]) -> tuple[Pattern, ...]:
    """Entries are strings or {pattern: ..., except_relations: [...]}; normalized like answers."""
    out: dict[str, Pattern] = {}
    for it in items or []:
        if isinstance(it, dict):
            text, exc = it.get("pattern", ""), it.get("except_relations", []) or []
        else:
            text, exc = it, []
        t = normalize_answer(str(text))
        if t:
            out[t] = Pattern(t, frozenset(str(r).strip().lower() for r in exc))
    return tuple(out.values())


@dataclass(frozen=True)
class RefusalPatterns:
    """Refusal patterns, normalized the same way as answers.

    exact:    the whole normalized answer equals the pattern ("Unknown." -> "unknown")
    contains: the pattern appears as a whole-word span ("X is not a person" -> "is not")
    A pattern can list relations where it is NOT a refusal ("none" for `religion`).
    An empty answer counts as abstain with pattern "<empty>".
    """

    exact: tuple[Pattern, ...]
    contains: tuple[Pattern, ...]

    @classmethod
    def from_lists(cls, exact: list[Any], contains: list[Any]) -> "RefusalPatterns":
        return cls(exact=_parse_patterns(exact), contains=_parse_patterns(contains))

    @classmethod
    def load(cls, path: str | Path) -> "RefusalPatterns":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls.from_lists(d.get("exact", []), d.get("contains", []))

    def find(self, pred_norm: str, relation: str | None = None) -> str | None:
        """The matching pattern, or None if the (normalized) answer is not a refusal."""
        if not pred_norm:
            return EMPTY_PATTERN
        rel = (relation or "").strip().lower()
        for p in self.exact:
            if pred_norm == p.text and rel not in p.except_relations:
                return p.text
        for p in self.contains:
            if span_in(p.text, pred_norm) and rel not in p.except_relations:
                return p.text
        return None


def auto_label(m: dict[str, Any], refusal: RefusalPatterns, relation: str | None = None) -> tuple[str, str | None]:
    """(label, refusal pattern): abstain > correct (strict) > partial (relaxed only) > wrong."""
    pattern = refusal.find(m["pred_norm"], relation)
    if pattern is not None:
        return "abstain", pattern
    if m["match_strict"]:
        return "correct", None
    if m["match_relaxed"]:
        return "partial", None
    return "wrong", None
