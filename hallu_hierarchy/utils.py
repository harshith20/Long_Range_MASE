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


def default_output_root() -> Path:
    """/kaggle/working/outputs on Kaggle, else hallu_hierarchy/outputs (gitignored)."""
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/outputs")
    return PKG_DIR / "outputs"


def run_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("output_root") or default_output_root())
    return root / cfg["run_name"]


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
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else None}


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
