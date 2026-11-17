"""Plain-text summary blocks: stdlib only, percentages next to every count, always show n."""

from __future__ import annotations

from typing import Any

WIDTH = 81
_PARTIAL = " ▏▎▍▌▋▊▉"


def pct(k: int, n: int) -> str:
    """'  34 (17.0%)'"""
    return f"{k:>4} ({100.0 * k / n:5.1f}%)" if n else f"{k:>4} (  n/a )"


def bar(frac: float, width: int = 20) -> str:
    """Unicode bar for a fraction in [0, 1], with 1/8-character resolution."""
    frac = min(max(frac, 0.0), 1.0)
    eighths = round(frac * width * 8)
    full, rest = divmod(eighths, 8)
    return "█" * full + (_PARTIAL[rest] if rest else "")


def model_short(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("id", "?")).split("/")[-1].lower().replace("-instruct", "")


def block(stage: str, cfg: dict[str, Any], n: int, lines: list[str]) -> str:
    title = f" STAGE: {stage} | {model_short(cfg)} | {cfg.get('data', {}).get('dataset', '?')} | n={n} "
    left = max((WIDTH - len(title)) // 2, 4)
    head = "=" * left + title + "=" * max(WIDTH - left - len(title), 4)
    return "\n".join([head, *lines, "=" * len(head)])


def kv(key: str, value: str, width: int = 26) -> str:
    return f"{key:<{width}}{value}"
