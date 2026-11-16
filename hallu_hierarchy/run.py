"""Entry point: python -m hallu_hierarchy.run --config configs/<file>.yaml [--stages a b]"""

from __future__ import annotations

import argparse
import importlib

from .utils import load_config, run_dir

# stage name -> module with a run(cfg) function (imported lazily so CPU-only stages
# don't need torch)
STAGES = {
    "data": "hallu_hierarchy.data.prepare",
    "generate": "hallu_hierarchy.generation.generate",
    "correctness": "hallu_hierarchy.labeling.correctness",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", choices=list(STAGES), help="override cfg['stages']")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    stages = args.stages or cfg.get("stages", list(STAGES))
    print(f"[run] {cfg['run_name']} -> {run_dir(cfg)}  stages={stages}")
    for name in stages:
        importlib.import_module(STAGES[name]).run(cfg)


if __name__ == "__main__":
    main()
