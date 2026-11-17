"""REPORT.md: one status line, automatic checks, key numbers, actions, and a block for Claude.

Each stage gets a StageReport from Report.stage(name) and fills:
  checks  hard (FAIL if not ok) or soft (WARN if not ok), each with a plain-words message
  key     the few lines the user reads
  action  rows needing a manual decision (the ACTION NEEDED section appears only if non-empty)
  claude  compact numbers for the FOR CLAUDE block
  plots   (title, draw(ax)) callables combined into DASHBOARD.png
run.py writes the report after the last stage, also when a stage crashes.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..utils import git_info, report_dir, run_dir, write_json
from .console import model_short

START = "==================== REPORT START ===================="
END = "==================== REPORT END ======================"
MAX_CLAUDE_LINES = 60
ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}


@dataclass
class Check:
    ok: bool
    msg: str
    hard: bool = False

    @property
    def state(self) -> str:
        return "pass" if self.ok else ("fail" if self.hard else "warn")


@dataclass
class StageReport:
    name: str
    n: int | None = None
    checks: list[Check] = field(default_factory=list)
    key: list[str] = field(default_factory=list)
    action: list[str] = field(default_factory=list)
    claude: list[str] = field(default_factory=list)
    plots: list[tuple[str, Callable[[Any], None]]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def hard(self, ok: bool, msg: str) -> None:
        self.checks.append(Check(bool(ok), msg, hard=True))

    def soft(self, ok: bool, msg: str) -> None:
        self.checks.append(Check(bool(ok), msg, hard=False))


class Report:
    def __init__(self, cfg: dict[str, Any], stages: list[str]) -> None:
        self.cfg, self.stages = cfg, stages
        self.parts: list[StageReport] = []
        self.t0 = time.perf_counter()
        self.git = git_info()

    def stage(self, name: str) -> StageReport:
        part = StageReport(name)
        part._t0 = time.perf_counter()  # type: ignore[attr-defined]
        self.parts.append(part)
        return part

    def finish_stage(self, part: StageReport) -> None:
        part.seconds = time.perf_counter() - part._t0  # type: ignore[attr-defined]

    def crashed(self, name: str, exc: BaseException) -> None:
        part = next((p for p in self.parts if p.name == name), None) or self.stage(name)
        self.finish_stage(part)
        part.hard(False, f"stage '{name}' crashed: {type(exc).__name__}: {exc}")
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        part.claude += ["traceback (last lines):"] + [
            "  " + ln for ln in "".join(tb).strip().splitlines()[-12:]
        ]

    # ------------------------------------------------------------------ render

    @property
    def checks(self) -> list[Check]:
        return [c for p in self.parts for c in p.checks]

    @property
    def status(self) -> str:
        states = {c.state for c in self.checks}
        return "fail" if "fail" in states else ("warn" if "warn" in states else "pass")

    def header(self) -> str:
        n = next((p.n for p in reversed(self.parts) if p.n is not None), None)
        dirty = " (uncommitted changes)" if self.git.get("dirty") else ""
        return " | ".join([
            f"Milestone: {self.cfg.get('milestone', '?')}",
            model_short(self.cfg),
            str(self.cfg.get("data", {}).get("dataset", "?")),
            f"n={n if n is not None else '?'}",
            f"stages: {','.join(self.stages)}",
            f"commit {self.git.get('short')}{dirty}",
        ])

    def claude_block(self) -> list[str]:
        cfg = self.cfg
        lines = [
            self.header(),
            f"status={self.status} run={cfg.get('run_name')} seed={cfg.get('seed')} "
            f"model={cfg.get('model', {}).get('id')} run_time={time.perf_counter() - self.t0:.0f}s",
            f"output={run_dir(cfg)}",
        ]
        if cfg.get("input_run_dir_used"):
            lines.append(f"inputs={cfg['input_run_dir_used']}")
        for p in self.parts:
            lines.append(f"[{p.name}] {p.seconds:.1f}s")
            lines += ["  " + ln for ln in p.claude]
        bad = [c for c in self.checks if not c.ok]
        lines.append("warnings/failures: " + ("none" if not bad else ""))
        lines += [f"  {ICON[c.state]} {c.msg}" for c in bad]
        if len(lines) > MAX_CLAUDE_LINES:
            cut = len(lines) - MAX_CLAUDE_LINES + 1
            lines = lines[: MAX_CLAUDE_LINES - 1] + [f"... {cut} lines cut (full numbers in outputs/<run>/summary.json)"]
        return lines

    def render(self) -> str:
        out = [START, f"STATUS: {ICON[self.status]} {self.status.upper()}", self.header(), "", "CHECKS"]
        out += [f"{ICON[c.state]} {c.msg}" for c in self.checks] or ["(no checks)"]
        out += ["", "KEY NUMBERS"]
        for p in self.parts:
            if p.key:
                out.append(f"  [{p.name}]")
                out += ["  " + ln for ln in p.key]
        actions = [ln for p in self.parts for ln in p.action]
        if actions:
            out += ["", "ACTION NEEDED"] + ["  " + ln for ln in actions]
        out += ["", "FOR CLAUDE (paste everything between the markers)"]
        out += ["  " + ln for ln in self.claude_block()]
        out.append(END)
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------ write

    def write(self) -> Path:
        rdir = report_dir(self.cfg)
        rdir.mkdir(parents=True, exist_ok=True)
        text = self.render()
        (rdir / "REPORT.md").write_text(text, encoding="utf-8")

        from .plots import dashboard  # matplotlib optional

        plots = [pl for p in self.parts for pl in p.plots]
        dash = dashboard(plots, rdir / "DASHBOARD.png", self.header())
        if dash is None and (rdir / "DASHBOARD.png").exists():
            (rdir / "DASHBOARD.png").unlink()  # don't leave a stale one from an earlier run

        write_json(run_dir(self.cfg) / "summary.json", {
            "status": self.status,
            "header": self.header(),
            "git": self.git,
            "checks": [{"ok": c.ok, "hard": c.hard, "msg": c.msg} for c in self.checks],
            "stages": {p.name: {"n": p.n, "seconds": p.seconds, **p.summary} for p in self.parts},
        })
        return rdir / "REPORT.md"
