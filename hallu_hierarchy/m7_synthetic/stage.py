"""`--stages m7`: generate the synthetic corpus, train a model from scratch, evaluate its checkpoints.

Runs inside the normal runner (hallu_hierarchy.run), so the Kaggle guards and REPORT.md of CLAUDE.md §12 apply.
Training and evaluation run as subprocesses (torchrun for 2 GPUs). The token corpus goes to a scratch folder
(not saved as Kaggle output); the metadata tables, checkpoints and evaluation results go to the run folder.

Config section (configs/m7_*.yaml):
    m7:
      generator_config: m7_synthetic/data/config_v1.yaml   # relative to hallu_hierarchy/
      fraction: 1.0           # share of people per cell (0.01 = smoke corpus)
      model: s124             # s124 | s25 | tiny
      batch: 32
      max_steps: null         # smoke runs only
      time_limit_h: 10.5      # save state and stop before Kaggle's 12 h limit
      states_ckpts: all       # hidden states for all checkpoints or only the last
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..utils import run_dir

HH = Path(__file__).resolve().parents[1]
KEEP = ["people.parquet", "exposure_log.parquet", "cooccurrence.parquet", "documents.parquet", "budget.json",
        "tokenizer.json", "templates.json", "config.yaml"]


def _sh(cmd: list[str]) -> None:
    print("[m7] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run(cfg: dict[str, Any], part: Any = None) -> None:
    from .data.generate import generate, load_config, load_templates

    m = cfg["m7"]
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    scratch = Path(m.get("scratch_dir") or ("/tmp/m7_data" if Path("/kaggle").exists() else rdir / "corpus"))
    t0 = time.time()

    # 1) corpus (deterministic from the generator config; regenerated in scratch if missing)
    gcfg = load_config(HH / m["generator_config"])
    if not (scratch / "budget.json").exists():
        generate(gcfg, load_templates(), scratch, m.get("fraction", 1.0))
    budget = json.loads((scratch / "budget.json").read_text())
    (rdir / "data").mkdir(exist_ok=True)
    for f in KEEP:
        if (scratch / f).exists():
            shutil.copy(scratch / f, rdir / "data" / f)
    t_gen = time.time() - t0

    # 2) training (one pass, stored order); resumes from state.pt if a previous session stopped
    train_dir = rdir / "train"
    import torch
    n_gpu = torch.cuda.device_count()
    args = ["-m", "hallu_hierarchy.m7_synthetic.train.train", "--data", str(scratch), "--out", str(train_dir),
            "--model", m.get("model", "s124"), "--batch", str(m.get("batch", 32)),
            "--time-limit-h", str(m.get("time_limit_h", 10.5)), "--seed", str(cfg.get("seed", 0))]
    if m.get("max_steps"):
        args += ["--max-steps", str(m["max_steps"])]
    if n_gpu > 1:
        _sh([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}", *args])
    else:
        _sh([sys.executable, *args])
    t_train = time.time() - t0 - t_gen
    done = (train_dir / "DONE").exists()

    # 3) evaluation of every checkpoint on qa_test people
    _sh([sys.executable, "-m", "hallu_hierarchy.m7_synthetic.eval.evaluate", "--data", str(scratch),
         "--run", str(train_dir), "--states-ckpts", m.get("states_ckpts", "all")])
    t_eval = time.time() - t0 - t_gen - t_train

    # 4) report
    meta = json.loads((train_dir / "train_meta.json").read_text())
    log = [json.loads(ln) for ln in (train_dir / "train_log.jsonl").read_text().splitlines() if ln.strip()]
    steps = [r for r in log if "loss" in r]
    ckpts = [r for r in log if "checkpoint" in r]
    tps = sorted(r["tok_per_s"] for r in steps[1:]) or [0]
    import pandas as pd
    res = pd.concat([pd.read_parquet(p) for p in sorted((train_dir / "eval").glob("ckpt_*/results.parquet"))])
    last = res[res.checkpoint == res.checkpoint.max()]
    ppl = pd.read_parquet(scratch / "people.parquet")
    lq = last[last.prompt == "qa"].merge(ppl[["person_id", "exp_birth"]], on="person_id")
    acc_hi = (lq[lq.exp_birth == lq.exp_birth.max()].label == "FA").mean()
    acc_lo = (lq[lq.exp_birth == 1].label == "FA").mean()
    if part is not None:
        part.n = int(last.person_id.nunique())
        part.hard(budget.get("cities_multi_token", 0) == 0, "every city is a single token (answer = one token)")
        part.hard(done, "training finished all steps" if done else "training stopped early (time limit): rerun resumes")
        part.hard(all(r["loss"] == r["loss"] for r in steps), "no NaN loss")
        part.soft(acc_hi > acc_lo, f"QA accuracy rises with exposure: exp {lq.exp_birth.max()} {acc_hi:.1%} vs exp 1 {acc_lo:.1%}"
                  " (plan §3 check 1)")
        part.key.append(f"corpus {budget['measured_this_run']['tokens_total'] / 1e6:.1f}M tokens | people "
                        f"{budget['measured_this_run']['n_people']} | generator {budget['generator_version']} | "
                        f"{t_gen / 60:.1f} min")
        part.key.append(f"model {meta['model']} ({meta['params'] / 1e6:.1f}M params, {meta['non_embedding_params'] / 1e6:.1f}M "
                        f"non-embedding) | GPUs {n_gpu} | {meta['amp']} | steps {meta['steps']} × {meta['tokens_per_step']} tokens")
        part.key.append(f"training {t_train / 3600:.2f} h | median {tps[len(tps) // 2]:,} tokens/s | final loss "
                        f"{steps[-1]['loss'] if steps else '?'} | evaluation {t_eval / 60:.1f} min")
        part.key.append("checkpoints (quick first-token accuracy on 500 qa_test people):")
        part.key += [f"  {r['checkpoint']}: {r['tokens_seen'] / 1e6:.0f}M tokens  QA {r['qa_acc']:.1%}  "
                     f"completion {r['completion_acc']:.1%}" for r in ckpts]
        summary = (train_dir / "eval" / "SUMMARY.md").read_text(encoding="utf-8")
        last_block = summary.split("\n## ")[-1]
        part.key += ["last checkpoint:"] + ["  " + ln for ln in last_block.splitlines()]
        part.claude.append(f"meta={json.dumps(meta)}")
        part.claude.append(f"t_gen={t_gen:.0f}s t_train={t_train:.0f}s t_eval={t_eval:.0f}s tok_per_s_median={tps[len(tps) // 2]}")
        part.claude += [json.dumps(r) for r in ckpts]
