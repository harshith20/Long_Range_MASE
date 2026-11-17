"""Shared helpers: config loading, output paths, seeds, meta.json, JSONL I/O."""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml

PKG_DIR = Path(__file__).resolve().parent


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config. Relative paths are tried as given, then relative to the package."""
    p = Path(path)
    if not p.exists():
        alt = PKG_DIR / path
        if not alt.exists():
            raise FileNotFoundError(f"Config not found: {path} (also tried {alt})")
        p = alt
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p.resolve())
    return cfg


KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")


def on_kaggle() -> bool:
    return KAGGLE_WORKING.is_dir()


def default_output_root() -> Path:
    """/kaggle/working/outputs on Kaggle, else hallu_hierarchy/outputs (gitignored)."""
    return KAGGLE_WORKING / "outputs" if on_kaggle() else PKG_DIR / "outputs"


def report_dir(cfg: dict[str, Any]) -> Path:
    """Where REPORT.md / DASHBOARD.png / REVIEW.csv go: /kaggle/working on Kaggle,
    else the local (gitignored) outputs folder."""
    if cfg.get("report_dir"):
        return Path(cfg["report_dir"])
    if cfg.get("output_root"):
        return Path(cfg["output_root"])
    return KAGGLE_WORKING if on_kaggle() else PKG_DIR / "outputs"


def run_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("output_root") or default_output_root())
    return root / cfg["run_name"]


def input_run_dirs(cfg: dict[str, Any]) -> list[Path]:
    """Run folders to read upstream outputs from: this run first, then `input_run_dir`.

    `input_run_dir` points at saved outputs (e.g. a Kaggle Dataset). The run folder may sit
    directly there, under <dir>/<run_name>, or under <dir>/outputs/<run_name>.
    """
    dirs = [run_dir(cfg)]
    extra = cfg.get("input_run_dir")
    if extra:
        base = Path(extra)
        for cand in (base, base / cfg["run_name"], base / "outputs" / cfg["run_name"]):
            if cand.is_dir() and cand not in dirs:
                dirs.append(cand)
    elif KAGGLE_INPUT.is_dir():
        dirs += [d for d in discover_kaggle_inputs(cfg["run_name"]) if d not in dirs]
    return dirs


def discover_kaggle_inputs(run_name: str, root: Path = KAGGLE_INPUT, max_depth: int = 6) -> list[Path]:
    """Folders named `run_name` (holding stage outputs) inside attached Kaggle inputs."""
    found = []
    for depth in range(1, max_depth + 1):
        for cand in sorted(root.glob("/".join(["*"] * (depth - 1) + [run_name]))):
            if cand.is_dir() and any((cand / s).is_dir() for s in ("generation", "data", "correctness")):
                found.append(cand)
    return found


def find_upstream(cfg: dict[str, Any], rel: str) -> Path | None:
    """First existing `<run folder>/<rel>` among input_run_dirs(cfg), or None."""
    for d in input_run_dirs(cfg):
        if (d / rel).exists():
            return d / rel
    return None


def resolve_pkg_path(path: str | os.PathLike | None) -> Path | None:
    """Absolute paths as given; relative paths relative to the package folder."""
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PKG_DIR / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_info() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PKG_DIR, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".")
    return {
        "commit": commit or "unknown",
        "short": (commit or "unknown")[:7],
        "dirty": bool(status) if status is not None else None,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_meta(out_dir: Path, cfg: dict[str, Any], stage: str, extra: dict[str, Any] | None = None) -> None:
    """Every output folder gets a meta.json (commit, model, dataset, seed, full config, time)."""
    meta = {
        "stage": stage,
        "git": git_info(),
        "model_id": cfg.get("model", {}).get("id"),
        "dataset": cfg.get("data", {}).get("dataset"),
        "seed": cfg.get("seed"),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_config_path"),
    }
    if extra:
        meta.update(extra)
    write_json(out_dir / "meta.json", meta)
