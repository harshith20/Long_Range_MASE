"""Entry point:
    python -m hallu_hierarchy.run --config configs/<file>.yaml --stages data,generation,correctness
                                  [--input-run-dir DIR]

Every run ends by writing REPORT.md (+ DASHBOARD.png) to /kaggle/working (locally: outputs/),
also when a stage crashes; the exit code is then 1.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback

from .reporting.report import Report
from .utils import load_config, run_dir

# stage name -> module with run(cfg, report) (imported lazily so CPU-only stages don't need torch)
STAGES = {
    "data": "hallu_hierarchy.data.prepare",
    "generation": "hallu_hierarchy.generation.generate",
    "subject_states": "hallu_hierarchy.generation.subject_states",
    "correctness": "hallu_hierarchy.correctness.stage",
}
ALIASES = {"generate": "generation"}


def parse_stages(values: list[str] | None) -> list[str] | None:
    """Accept '--stages a b', '--stages a,b' or a mix; old name 'generate' still works."""
    if not values:
        return None
    names = [ALIASES.get(s.strip(), s.strip()) for v in values for s in v.split(",") if s.strip()]
    unknown = [s for s in names if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", help=f"override cfg['stages']; from {list(STAGES)}")
    ap.add_argument(
        "--input-run-dir",
        help="saved outputs to read upstream stages from. On Kaggle, attached inputs are "
        "searched automatically for a folder named <run_name>.",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.input_run_dir:
        cfg["input_run_dir"] = args.input_run_dir
    stages = parse_stages(args.stages) or parse_stages(cfg.get("stages")) or list(STAGES)
    print(f"[run] {cfg['run_name']} -> {run_dir(cfg)}  stages={stages}")

    report = Report(cfg, stages)
    code = 0
    for name in stages:
        part = report.stage(name)
        try:
            importlib.import_module(STAGES[name]).run(cfg, part)
            report.finish_stage(part)
        except Exception as exc:  # report the crash, skip later stages
            traceback.print_exc()
            report.crashed(name, exc)
            code = 1
            break
    path = report.write()
    print(f"[run] status={report.status.upper()} -> {path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
