"""Feature functions for the entropy hierarchy (CLAUDE.md §3). Pure functions, unit-tested.

Conventions
  * "content tokens": generated tokens without the final EOS (EOS is a stop decision, not
    answer content). If the answer is only EOS, the EOS token is used.
  * Entropies in nats.
  * Hidden-state index 0 = embeddings, index 36 = final layer AFTER the final norm (verified in
    M2 prep: the lens at index 36 reproduces the real first-token distribution, E10).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable

import numpy as np

from ..data.normalize import normalize_answer


def content_slice(values: list[float], ends_with_eos: bool) -> list[float]:
    return values[:-1] if ends_with_eos and len(values) > 1 else values


def token_features(greedy: dict) -> dict[str, float]:
    """Mean and max per-token entropy of the greedy answer (content tokens)."""
    h = content_slice(greedy["token_entropy"], greedy["ends_with_eos"])
    return {"tok_mean": float(np.mean(h)), "tok_max": float(np.max(h))}


def sequence_nll(samples: list[dict]) -> float:
    """Mean over samples of the length-normalized negative log-likelihood (content tokens)."""
    vals = []
    for s in samples:
        lp = content_slice(s["token_logprob"], s["ends_with_eos"])
        if lp:
            vals.append(-float(np.mean(lp)))
    return float(np.mean(vals)) if vals else float("nan")


def unique_answers(texts: list[str], abstain: list[bool]) -> tuple[list[str], list[int]]:
    """Distinct normalized non-abstaining answers (first spelling kept) and, per sample, the index
    of its distinct answer (-1 for abstaining samples)."""
    keys: dict[str, int] = {}
    reps: list[str] = []
    idx = []
    for t, a in zip(texts, abstain):
        if a:
            idx.append(-1)
            continue
        k = normalize_answer(t)
        if k not in keys:
            keys[k] = len(reps)
            reps.append(t)
        idx.append(keys[k])
    return reps, idx


def cluster_by_entailment(n_unique: int, entails: Callable[[int, int], bool]) -> list[int]:
    """Greedy bidirectional-entailment clustering (Kuhn et al. 2023) over distinct answers.
    Answer j joins the first existing cluster whose representative r satisfies
    entails(j, r) and entails(r, j); otherwise it starts a new cluster."""
    reps: list[int] = []
    cluster_of = []
    for j in range(n_unique):
        for c, r in enumerate(reps):
            if entails(j, r) and entails(r, j):
                cluster_of.append(c)
                break
        else:
            reps.append(j)
            cluster_of.append(len(reps) - 1)
    return cluster_of


def semantic_entropy(sample_answer_idx: list[int], answer_cluster: list[int]) -> tuple[float, int, list[int]]:
    """Entropy over cluster frequencies of the N samples. Abstaining samples (-1) form ONE extra
    cluster of their own. Returns (entropy, number of clusters, cluster id per sample)."""
    abstain_id = (max(answer_cluster) + 1) if answer_cluster else 0
    per_sample = [abstain_id if i < 0 else answer_cluster[i] for i in sample_answer_idx]
    n = len(per_sample)
    counts = Counter(per_sample)
    h = -sum((c / n) * math.log(c / n) for c in counts.values()) if n else float("nan")
    return float(h) + 0.0, len(counts), per_sample


def lens_features(lens_entropy: np.ndarray, lens_top1: np.ndarray, layers: list[int]) -> dict[str, float]:
    """Logit-lens entropy at the chosen hidden-state indices, and l* = first index where the final
    answer's first token is top-1 (len(lens_top1) if never)."""
    out = {f"lens_H_{i}": float(lens_entropy[i]) for i in layers}
    hits = np.flatnonzero(lens_top1)
    out["lens_lstar"] = float(hits[0]) if hits.size else float(len(lens_top1))
    return out


def attention_features(heads: np.ndarray, layers: list[int]) -> dict[str, float]:
    """Attention mass of the last prompt token on the subject tokens, mean and max over heads,
    at the chosen attention-layer indices (0-based transformer blocks). NaN if no subject."""
    out = {}
    for layer in layers:
        row = heads[layer]
        out[f"attn_mean_{layer}"] = float(np.mean(row))
        out[f"attn_max_{layer}"] = float(np.max(row))
    return out
