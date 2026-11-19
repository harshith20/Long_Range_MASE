"""Metrics with numpy only (no sklearn), so every stage can report them."""

from __future__ import annotations

import numpy as np


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """P(score of a random positive > score of a random negative), ties count 1/2 (Mann–Whitney U).
    NaN scores are dropped. None if either class is empty."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    keep = ~np.isnan(score)
    score, positive = score[keep], positive[keep]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = _avg_ranks(score)
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman_matrix(cols: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Spearman correlation between columns (rows with any NaN dropped)."""
    names = list(cols)
    m = np.column_stack([np.asarray(cols[k], dtype=np.float64) for k in names])
    m = m[~np.isnan(m).any(axis=1)]
    ranks = np.column_stack([_avg_ranks(m[:, j]) for j in range(m.shape[1])])
    return names, np.corrcoef(ranks, rowvar=False)


def vif_from_corr(corr: np.ndarray) -> np.ndarray:
    """Variance inflation factors = diagonal of the inverse correlation matrix."""
    return np.diag(np.linalg.pinv(corr))
