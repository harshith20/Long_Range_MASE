"""Entry point: python -m hallu_hierarchy.run --config configs/<file>.yaml [--stages a b]
                                            [--input-run-dir DIR]"""

from __future__ import annotations

import argparse
import importlib

from .utils import load_config, run_dir

# stage name -> module with a run(cfg) function (imported lazily so CPU-only stages
# don't need torch)
STAGES = {
    "data": "hallu_hierarchy.data.prepare",
    "generate": "hallu_hierarchy.generation.generate",
    "correctness": "hallu_hierarchy.correctness.stage",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", choices=list(STAGES), help="override cfg['stages']")
    ap.add_argument(
        "--input-run-dir",
        help="saved outputs to read upstream stages from (e.g. /kaggle/input/<dataset>); "
        "overrides cfg['input_run_dir']",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.input_run_dir:
        cfg["input_run_dir"] = args.input_run_dir
    stages = args.stages or cfg.get("stages", list(STAGES))
    print(f"[run] {cfg['run_name']} -> {run_dir(cfg)}  stages={stages}")
    for name in stages:
        importlib.import_module(STAGES[name]).run(cfg)


if __name__ == "__main__":
    main()
