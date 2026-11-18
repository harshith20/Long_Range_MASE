"""Tests for scripts/kaggle_run.sh pieces that run locally (no Kaggle calls)."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

from hallu_hierarchy.utils import PKG_DIR

SCRIPT = PKG_DIR / "scripts" / "kaggle_run.sh"


def _build_block() -> str:
    """The Python heredoc that builds the push folder."""
    text = SCRIPT.read_text()
    m = re.search(r'^"\$PY" - "\$META".*?<<\'PYEOF\'\n(.*?)^PYEOF$', text, re.S | re.M)
    assert m, "build block not found in kaggle_run.sh"
    return m.group(1)


@pytest.mark.parametrize("gpu", ["on", "off"])
def test_injected_notebook_first_cell_executes(tmp_path, gpu):
    """The injected RUN_REQUEST line must be valid Python (version 3 failed on `false`)."""
    (tmp_path / "inject.py").write_text(_build_block())
    build = tmp_path / "build"
    build.mkdir()
    subprocess.run(
        [sys.executable, str(tmp_path / "inject.py"), str(PKG_DIR / "kaggle" / "kernel-metadata.json"),
         str(PKG_DIR / "kaggle" / "run.ipynb"), str(build), gpu, "abc1234", "configs/x.yaml",
         "correctness", "20261118T000000Z"],
        check=True,
    )
    nb = json.loads((build / "run.ipynb").read_text())
    first_line = "".join(nb["cells"][1]["source"]).splitlines()[0]
    ns: dict = {}
    exec(first_line, {}, ns)  # what Kaggle runs first
    assert ns["RUN_REQUEST"] == {"commit": "abc1234", "config": "configs/x.yaml", "stages": "correctness",
                                 "gpu": gpu == "on", "requested_utc": "20261118T000000Z"}
    meta = json.loads((build / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] == ("true" if gpu == "on" else "false")
    assert "title" not in meta  # an existing kernel must not be renamed


def test_committed_notebook_has_one_marker():
    nb = json.loads((PKG_DIR / "kaggle" / "run.ipynb").read_text())
    hits = [c for c in nb["cells"] if c["cell_type"] == "code"
            and "".join(c["source"]).startswith("RUN_REQUEST = None")]
    assert len(hits) == 1


@pytest.mark.parametrize(
    "args,msg",
    [
        ([], "--stages is required"),
        (["--stages", "correctness"], "--gpu on|off is required"),
        (["--stages", "correctness", "--gpu", "on"], "CPU-only"),
        (["--stages", "subject_states", "--gpu", "off"], "need the GPU"),
        (["--stages", "generation", "--gpu", "on"], "--allow-generation"),
        (["--stages", "entropyy", "--gpu", "on"], "unknown stage"),
    ],
)
def test_guards_refuse(args, msg):
    if not (PKG_DIR / ".venv" / "bin" / "python").exists():
        pytest.skip("local Kaggle CLI venv not present (e.g. on Kaggle)")
    r = subprocess.run([str(SCRIPT), *args], capture_output=True, text=True)
    assert r.returncode == 2 and msg in r.stderr, r.stderr
