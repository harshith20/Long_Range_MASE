"""Pure co-occurrence logic for M3 (no I/O, unit-tested).

A wrong answer is an *associated* hallucination (AH) when the model's answer co-occurs with the
subject in the corpus (here: Wikipedia, the Qwen proxy), and *unassociated* (UH) when it does not.

Signals per wrong answer
  cooc_page    the answer appears on the subject's own Wikipedia page (whole-word, normalized)
  linked_hits  pages that contain the answer AND link to the subject's page (`linksto:` search)
  cooc_count   linked_hits + cooc_page
Rules (the cutoff is a parameter, CLAUDE.md §9): "page" = cooc_page; "count>=k" = cooc_count >= k.
"""

from __future__ import annotations

from typing import Any

from ..correctness.matching import span_in, strip_head_words


def answer_candidates(pred_norm: str, head_words: tuple[str, ...] = ()) -> list[str]:
    """The normalized answer and its head-word-stripped copy ('action film' -> 'action')."""
    cands = [pred_norm, strip_head_words(pred_norm, head_words) if head_words else None]
    return [c for c in dict.fromkeys(cands) if c and len(c) >= 2]


def any_on(text: str, cands: list[str]) -> bool:
    return bool(text) and any(span_in(c, text) for c in cands)


SENTENCE_WORDS = frozenset({"is", "was", "are", "were", "has", "had", "have", "not", "no", "none", "known",
                            "unknown", "born", "died", "does", "did", "unspecified", "specified"})


def answer_status(pred_norm: str, max_words: int = 12) -> str:
    """'ok' | 'empty' | 'sentence'.

    'sentence': a clause ("remco campert has no known father", "paris was not born in any city") or very
    long. These are mostly refusals the pattern list missed; matching them on the subject page would
    find the subject's own name and call them AH, so they are left unscored and listed instead.
    Long names ("thomas howard 3rd duke of norfolk") stay 'ok': they contain no sentence words.
    """
    if len(pred_norm) < 2:
        return "empty"
    words = pred_norm.split()
    if len(words) >= max_words or (len(words) >= 3 and SENTENCE_WORDS & set(words)):
        return "sentence"
    return "ok"


def rule_names(cutoffs: list[int]) -> list[str]:
    return ["page"] + [f"count>={k}" for k in cutoffs]


def is_ah(rule: str, cooc_page: bool, cooc_count: int | None) -> bool | None:
    """AH under one rule; None if the rule needs counts that were not computed."""
    if rule == "page":
        return bool(cooc_page)
    k = int(rule.split(">=")[1])
    return None if cooc_count is None else cooc_count >= k


def final_label(row: dict[str, Any], rule: str, ambiguity_rule: str) -> str:
    """Three-class label used downstream (FA / AH / UH) or the reason a row is left out.

    FA = correct (training-eligible); AH / UH = eligible wrong, scored; left out: abstain, partial,
    gold_noise ('excluded'), unscored wrong answers ('unscored'), and AH rows whose subject is
    ambiguous under `ambiguity_rule` ('ambiguous_confirmed', CLAUDE.md §3).
    """
    lab = row["label"]
    if lab == "abstain":
        return "abstain"
    if row["exclude_from_training"]:
        return "excluded"
    if lab == "correct":
        return "FA"
    if lab != "wrong":
        return "excluded"
    if row["m3_status"] != "ok":
        return "unscored"
    ah = row["ah"].get(rule)
    if ah is None:
        return "unscored"
    if not ah:
        return "UH"
    if row.get(ambiguity_rule):
        return "ambiguous_confirmed"
    return "AH"
