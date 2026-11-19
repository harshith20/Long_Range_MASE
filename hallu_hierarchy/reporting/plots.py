"""DASHBOARD.png: all plots of a run in one image. matplotlib is optional (preinstalled on
Kaggle); without it the dashboard is skipped — every important number is in REPORT.md."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

COLORS = {"correct": "#2e7d32", "wrong": "#c62828", "abstain": "#9e9e9e", "partial": "#f9a825"}


def dashboard(plots: list[tuple[str, Callable[[Any], None]]], path: Path, title: str) -> Path | None:
    if not plots:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    ncol = min(2, len(plots))
    nrow = math.ceil(len(plots) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow), squeeze=False)
    for ax, (name, draw) in zip(axes.flat, plots):
        draw(ax)
        ax.set_title(name)
    for ax in list(axes.flat)[len(plots):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def outcomes_by_relation(rows: list[dict[str, Any]]) -> Callable[[Any], None]:
    """Horizontal stacked bars: label counts per relation (labels present in the rows)."""

    def draw(ax: Any) -> None:
        rs = sorted(rows, key=lambda r: r["n"])
        names = [f"{r['relation']} (n={r['n']})" for r in rs]
        left = [0] * len(rs)
        for lab in ("correct", "wrong", "abstain", "partial"):
            vals = [r.get(lab, 0) for r in rs]
            ax.barh(names, vals, left=left, color=COLORS[lab], label=lab)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_xlabel("questions")
        ax.legend(loc="lower right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    return draw


def histogram(values: list[float], xlabel: str, bins: int = 30) -> Callable[[Any], None]:
    def draw(ax: Any) -> None:
        ax.hist(values, bins=bins, color="#1565c0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("questions")

    return draw


def histograms_by_label(rows: list[dict[str, Any]], key: str, bins: int = 30) -> Callable[[Any], None]:
    """Overlaid histograms of one feature for correct vs wrong (training-eligible rows)."""

    def draw(ax: Any) -> None:
        import math as _m

        for lab in ("correct", "wrong"):
            vals = [r[key] for r in rows if r["label"] == lab and not r.get("exclude_from_training")
                    and r[key] is not None and not _m.isnan(r[key])]
            if vals:
                ax.hist(vals, bins=bins, alpha=0.5, density=True, color=COLORS[lab], label=f"{lab} (n={len(vals)})")
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    return draw
