"""Linear probes with subject-grouped nested cross-validation (CLAUDE.md §3 Probes, Metrics).

For every outer fold f (train = the other folds, test = fold f):
  1. the binary target is made from the training rows only (SEP: entropy split at the point that
     minimizes the within-group squared error; familiarity: s_pop above the training median);
  2. the training rows are split again by subject (inner split); every (position, layer) source and
     every C in the grid is fitted on the inner-train part and scored (AUROC on the probe's own target)
     on the inner-validation part;
  3. for every source the best C is refitted on all training rows and scores the test fold
     -> one out-of-fold score per source (the "by layer" table);
  4. the source with the best inner AUROC is the fold's selected probe -> one out-of-fold score per row.
Nothing about the test fold is used for any choice. Hallucination labels are never a probe target,
except for the direct correctness probe (a baseline), whose target is correct vs wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..eval.metrics import auroc


# ----------------------------------------------------------------------------- targets

def best_split_threshold(values: np.ndarray) -> float:
    """Threshold t minimizing the summed squared error of {v <= t} and {v > t} around their means
    (the SEP binarization). Only splits between distinct values are considered."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[~np.isnan(v)]
    if v.size < 2 or v[0] == v[-1]:
        return float(v[0]) if v.size else float("nan")
    n = v.size
    cs, cs2 = np.cumsum(v), np.cumsum(v * v)
    k = np.arange(1, n)                                   # left group = first k values
    sse_l = cs2[:-1] - cs[:-1] ** 2 / k
    sse_r = (cs2[-1] - cs2[:-1]) - (cs[-1] - cs[:-1]) ** 2 / (n - k)
    sse = sse_l + sse_r
    sse[v[1:] == v[:-1]] = np.inf                         # cannot split inside a tie
    i = int(np.argmin(sse))
    return float((v[i] + v[i + 1]) / 2.0)


def binarize(values: np.ndarray, train: np.ndarray, rule: str) -> tuple[np.ndarray, float]:
    """1 = 'high' (high entropy / popular subject); the threshold comes from the training rows only."""
    if rule == "best_split":
        t = best_split_threshold(values[train])
    elif rule == "median":
        t = float(np.nanmedian(values[train]))
    else:
        raise ValueError(f"unknown binarization rule {rule!r}")
    return (values > t).astype(np.int8), t


def inner_split(train_idx: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows by subject: `frac` of the subjects go to the inner-validation part."""
    subj = np.unique(groups[train_idx])
    rng = np.random.default_rng(seed)
    val_subj = rng.choice(subj, size=max(1, int(round(frac * len(subj)))), replace=False)
    is_val = np.isin(groups[train_idx], val_subj)
    return train_idx[~is_val], train_idx[is_val]


# ----------------------------------------------------------------------------- fitting

def _standardize(X_tr: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return [(X - mu) / sd for X in (X_tr, *others)]


def fit_scores(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: list[np.ndarray], C_grid: list[float],
               max_iter: int, seed: int) -> list[list[np.ndarray]]:
    """Standardized logistic regression (balanced class weights) for each C, ascending with warm start.
    Returns, per C, the P(y=1) scores for each array in X_ev."""
    from sklearn.linear_model import LogisticRegression

    Z = _standardize(X_tr, *X_ev)
    m = LogisticRegression(class_weight="balanced", max_iter=max_iter, warm_start=True, random_state=seed)
    out = []
    for C in sorted(C_grid):
        m.set_params(C=C)
        m.fit(Z[0], y_tr)
        out.append([m.predict_proba(z)[:, 1] for z in Z[1:]])
    return out


# ----------------------------------------------------------------------------- nested CV

@dataclass
class ProbeResult:
    name: str
    sources: list[str]
    y: np.ndarray                                   # fold-specific binary target (-1 = not used)
    thresholds: dict[int, float]
    oof: dict[str, np.ndarray]                      # per source, out-of-fold P(high); NaN = not used
    inner_auc: dict[str, dict[int, float | None]]   # per source, per fold (best C)
    best_C: dict[str, dict[int, float]]
    selected: dict[int, str]                        # per fold, the source with the best inner AUROC
    oof_selected: np.ndarray
    seconds: float = 0.0
    n_fits: int = 0
    notes: list[str] = field(default_factory=list)


def run_probe(
    name: str,
    sources: dict[str, Callable[[], np.ndarray]],
    target: np.ndarray,
    use: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    rule: str | None,
    pcfg: dict[str, Any],
    seed: int,
) -> ProbeResult:
    """`sources` maps a name (e.g. 'tbg@24') to a loader of the [N, d] features (rows with NaN are
    dropped from `use`). `target` is continuous (binarized per fold with `rule`) or already 0/1 (rule None)."""
    t0 = time.time()
    n = len(target)
    fold_ids = sorted(int(f) for f in np.unique(folds[use]))
    y = np.full(n, -1, dtype=np.int8)
    thresholds: dict[int, float] = {}
    targets: dict[int, np.ndarray] = {}
    for f in fold_ids:
        train = use & (folds != f)
        if rule is None:
            yf, t = target.astype(np.int8), float("nan")
        else:
            yf, t = binarize(target, train, rule)
        targets[f], thresholds[f] = yf, t
        y[use & (folds == f)] = yf[use & (folds == f)]

    C_grid = [float(c) for c in pcfg["C_grid"]]
    oof: dict[str, np.ndarray] = {}
    inner_auc: dict[str, dict[int, float | None]] = {}
    best_C: dict[str, dict[int, float]] = {}
    n_fits = 0
    for src, load in sources.items():
        X = load()
        ok = use & ~np.isnan(X).any(axis=1)
        oof[src] = np.full(n, np.nan)
        inner_auc[src], best_C[src] = {}, {}
        for f in fold_ids:
            yf = targets[f]
            tr = np.flatnonzero(ok & (folds != f))
            te = np.flatnonzero(ok & (folds == f))
            if len(np.unique(yf[tr])) < 2 or len(te) == 0:
                inner_auc[src][f] = None
                continue
            itr, iva = inner_split(tr, groups, float(pcfg["inner_frac"]), seed + 1000 * f)
            aucs: list[float | None] = []
            if len(np.unique(yf[itr])) == 2 and len(np.unique(yf[iva])) == 2:
                per_C = fit_scores(X[itr], yf[itr], [X[iva]], C_grid, int(pcfg["max_iter"]), seed)
                aucs = [auroc(s[0], yf[iva] == 1) for s in per_C]
                n_fits += len(C_grid)
            ranked = sorted(C_grid)
            valid = [(a, c) for a, c in zip(aucs, ranked) if a is not None]
            C = max(valid)[1] if valid else ranked[len(ranked) // 2]
            inner_auc[src][f] = max(valid)[0] if valid else None
            best_C[src][f] = C
            oof[src][te] = fit_scores(X[tr], yf[tr], [X[te]], [C], int(pcfg["max_iter"]), seed)[0][0]
            n_fits += 1
        del X
        print(f"[probes] {name} {src}: inner AUROC by fold "
              f"{ {f: (None if a is None else round(a, 3)) for f, a in inner_auc[src].items()} }")

    selected: dict[int, str] = {}
    oof_sel = np.full(n, np.nan)
    for f in fold_ids:
        cand = [(a, s) for s in sources if (a := inner_auc[s].get(f)) is not None]
        if not cand:
            continue
        s = max(cand)[1]
        selected[f] = s
        m = folds == f
        oof_sel[m] = oof[s][m]
    return ProbeResult(name, list(sources), y, thresholds, oof, inner_auc, best_C, selected, oof_sel,
                       seconds=time.time() - t0, n_fits=n_fits)
