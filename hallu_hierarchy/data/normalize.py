"""Answer normalization (SQuAD-style + accent folding)."""

from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—", "…"}


def fold_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_answer(text: str) -> str:
    """Lowercase, fold accents, drop punctuation and articles, collapse whitespace."""
    text = fold_accents(text.lower())
    # Replace punctuation with spaces so "New-York" -> "new york" rather than "newyork".
    text = "".join(" " if c in _PUNCT else c for c in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())
