"""Chunked checkpointing for generation so Kaggle sessions can die and resume.

Chunk c covers questions [c*chunk_size, (c+1)*chunk_size). Each finished chunk is written as
shard_XXXXX.npz (arrays) + shard_XXXXX.jsonl (records). The .jsonl is written last and
marks the chunk as complete. A fingerprint guards against mixing shards from different
configs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ShardStore:
    def __init__(self, root: Path, chunk_size: int, fp: str) -> None:
        self.root, self.chunk_size, self.fp = Path(root), chunk_size, fp
        self.root.mkdir(parents=True, exist_ok=True)
        fp_file = self.root / "fingerprint.json"
        if fp_file.exists():
            old = json.loads(fp_file.read_text())["fingerprint"]
            if old != fp:
                raise RuntimeError(
                    f"{self.root} holds shards from a different config (fingerprint {old} != {fp}). "
                    "Use a new run_name or delete the shards."
                )
        else:
            fp_file.write_text(json.dumps({"fingerprint": fp}))

    def n_chunks(self, n_items: int) -> int:
        return math.ceil(n_items / self.chunk_size)

    def chunk_range(self, c: int, n_items: int) -> range:
        return range(c * self.chunk_size, min((c + 1) * self.chunk_size, n_items))

    def _paths(self, c: int) -> tuple[Path, Path]:
        return self.root / f"shard_{c:05d}.npz", self.root / f"shard_{c:05d}.jsonl"

    def is_done(self, c: int) -> bool:
        npz, jl = self._paths(c)
        return npz.exists() and jl.exists()

    def missing_chunks(self, n_items: int) -> list[int]:
        return [c for c in range(self.n_chunks(n_items)) if not self.is_done(c)]

    def save_chunk(self, c: int, records: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
        npz, jl = self._paths(c)
        tmp = npz.with_suffix(".npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, npz)
        tmp = jl.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, jl)  # completion marker

    def load_chunk(self, c: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        npz, jl = self._paths(c)
        with open(jl, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        return records, arrays

    def import_from(self, dirs: list[str | Path]) -> int:
        """Copy finished shards from earlier runs (e.g. a Kaggle input dataset). Returns count."""
        copied = 0
        for d in dirs:
            src = self._resolve_shard_dir(Path(d))
            if src is None:
                print(f"[resume] no shards found under {d}; skipping")
                continue
            other = json.loads((src / "fingerprint.json").read_text())["fingerprint"]
            if other != self.fp:
                raise RuntimeError(f"[resume] {src} fingerprint {other} != current {self.fp}")
            for jl in sorted(src.glob("shard_*.jsonl")):
                c = int(jl.stem.split("_")[1])
                npz = jl.with_suffix(".npz")
                if npz.exists() and not self.is_done(c):
                    dst_npz, dst_jl = self._paths(c)
                    shutil.copyfile(npz, dst_npz)
                    shutil.copyfile(jl, dst_jl)
                    copied += 1
        return copied

    @staticmethod
    def _resolve_shard_dir(d: Path) -> Path | None:
        for cand in (d, d / "shards", d / "generation" / "shards"):
            if (cand / "fingerprint.json").exists():
                return cand
        return None
