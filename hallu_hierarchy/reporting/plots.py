"""PNG plots to outputs/<run>/plots/. matplotlib is optional (preinstalled on Kaggle);
without it plots are skipped and only the console block + summary.json are written."""

from __future__ import annotations

from pathlib import Path
from typing import Any

COLORS = {"correct": "#2e7d32", "wrong": "#c62828", "abstain": "#9e9e9e"}


def outcomes_by_relation(rows: list[dict[str, Any]], path: Path, title: str) -> Path | None:
    """Horizontal stacked bars: correct / wrong / abstain counts per relation."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["n"])
    names = [f"{r['relation']} (n={r['n']})" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(rows) + 1.5))
    left = [0] * len(rows)
    for lab in ("correct", "wrong", "abstain"):
        vals = [r[lab] for r in rows]
        ax.barh(names, vals, left=left, color=COLORS[lab], label=lab)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("questions")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
