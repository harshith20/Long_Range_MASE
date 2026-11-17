"""Plain-text formatting helpers: stdlib only, percentages next to every count."""

from __future__ import annotations

from typing import Any

_PARTIAL = " ▏▎▍▌▋▊▉"


def pct(k: int, n: int) -> str:
    """'34 (17.0%)'"""
    return f"{k} ({100.0 * k / n:.1f}%)" if n else f"{k} (n/a)"


def bar(frac: float, width: int = 20) -> str:
    """Unicode bar for a fraction in [0, 1], with 1/8-character resolution."""
    frac = min(max(frac, 0.0), 1.0)
    eighths = round(frac * width * 8)
    full, rest = divmod(eighths, 8)
    return "█" * full + (_PARTIAL[rest] if rest else "")


def model_short(cfg: dict[str, Any]) -> str:
    """'Qwen/Qwen2.5-3B-Instruct' -> 'qwen2.5-3b'"""
    return str(cfg.get("model", {}).get("id", "?")).split("/")[-1].lower().replace("-instruct", "")


def table(rows: list[list[Any]], indent: str = "  ") -> list[str]:
    """Left-align the first column, right-align the rest."""
    if not rows:
        return []
    cols = list(zip(*[[str(c) for c in r] for r in rows]))
    widths = [max(len(c) for c in col) for col in cols]
    out = []
    for r in rows:
        cells = [str(c) for c in r]
        line = cells[0].ljust(widths[0]) + "".join("  " + c.rjust(w) for c, w in zip(cells[1:], widths[1:]))
        out.append(indent + line.rstrip())
    return out
