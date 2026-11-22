"""Metrics with numpy only (no sklearn), so every stage can report them."""

from __future__ import annotations

import numpy as np


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (vectorized: the bootstrap calls this thousands of times)."""
    x = np.asarray(x)
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    edges = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1], True])   # start of each tie group + end
    avg = (edges[:-1] + edges[1:] - 1) / 2.0 + 1.0                 # mean 1-based rank of each group
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.repeat(avg, np.diff(edges))
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


def bootstrap_auroc_ci(
    score: np.ndarray, positive: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """95% CI of the AUROC, resampling whole groups (subjects) with replacement (CLAUDE.md §3 Metrics)."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    keep = ~np.isnan(score)
    score, positive, groups = score[keep], positive[keep], np.asarray(groups)[keep]
    if n_boot <= 0 or positive.all() or not positive.any():
        return None, None
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = np.arange(len(score))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        times = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))  # draws per group
        idx = np.repeat(rows, times[inv])                             # every row of a drawn group, per draw
        a = auroc(score[idx], positive[idx])
        if a is not None:
            vals.append(a)
    if not vals:
        return None, None
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


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
