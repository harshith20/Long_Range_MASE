"""CPU tests for M4 probes: SEP split, inner split, nested CV, the stage end to end on synthetic states."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from hallu_hierarchy.eval.metrics import _avg_ranks, auroc, bootstrap_auroc_ci
from hallu_hierarchy.probes.core import best_split_threshold, binarize, inner_split, run_probe

PC = {"C_grid": [0.01, 0.1, 1.0], "inner_frac": 0.25, "max_iter": 300}


def test_avg_ranks_ties():
    assert _avg_ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75


def test_bootstrap_ci_contains_point_and_groups():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.3
    s = y + rng.standard_normal(400)
    lo, hi = bootstrap_auroc_ci(s, y, np.arange(400) // 2, n_boot=200)
    assert lo < auroc(s, y) < hi and hi - lo < 0.2


def test_best_split_separates_two_modes():
    v = np.r_[np.zeros(50), np.full(10, 0.1), np.full(40, 2.0), np.full(5, 2.2)]
    t = best_split_threshold(v)
    assert 0.1 < t < 2.0
    assert best_split_threshold(np.zeros(5)) == 0.0          # constant: no split


def test_binarize_uses_training_rows_only():
    v = np.array([0.0, 0.0, 5.0, 5.0, 100.0, 100.0])
    train = np.array([True, True, True, True, False, False])
    y, t = binarize(v, train, "best_split")
    assert t == 2.5 and y.tolist() == [0, 0, 1, 1, 1, 1]


def test_inner_split_is_subject_grouped():
    groups = np.repeat(np.arange(20), 3)
    tr, va = inner_split(np.arange(60), groups, 0.25, seed=0)
    assert set(groups[tr]).isdisjoint(groups[va]) and len(set(groups[va])) == 5


def test_run_probe_finds_informative_layer():
    rng = np.random.default_rng(0)
    n, d = 400, 16
    target = rng.random(n)
    good = rng.standard_normal((n, d)).astype(np.float32)
    good[:, 0] += 4 * (target > 0.5)                          # layer "b" encodes the target
    noise = rng.standard_normal((n, d)).astype(np.float32)
    folds = np.arange(n) % 5
    res = run_probe("t", {"a": lambda: noise, "b": lambda: good}, target, np.ones(n, bool), folds,
                    np.arange(n), "median", PC, seed=0)
    assert set(res.selected.values()) == {"b"}
    assert auroc(res.oof_selected, res.y == 1) > 0.9
    assert not np.isnan(res.oof_selected).any()


# ----------------------------------------------------------------------------- stage

def _fixture(tmp_path, n=300, with_m3=True):
    run = tmp_path / "out" / "r1"
    for d in ("generation", "correctness", "subject_states", "entropy", "labeling"):
        (run / d).mkdir(parents=True)
    rng = np.random.default_rng(1)
    layers = [0, 4, 8]
    wrong = rng.random(n) < 0.6
    s_pop = rng.integers(1, 10000, n)
    gens, labels, feats, m3 = [], [], [], []
    for i in range(n):
        q = f"q{i}"
        rel = ["capital", "director", "genre"][i % 3]
        lab = "wrong" if wrong[i] else "correct"
        if i % 50 == 7:
            lab = "abstain"
        gens.append({"qid": q})
        labels.append({"qid": q, "label": lab, "fold": i % 5, "relation": rel,
                       "relation_group": "closed" if rel != "director" else "open", "subject_id": str(i),
                       "s_pop": int(s_pop[i]), "flags": {"gold_noise": False},
                       "exclude_from_training": lab == "abstain"})
        sem = float(wrong[i]) * 1.5 + rng.random() * 0.3
        feats.append({"qid": q, "fold": i % 5, "sem_entropy": sem, "seq_nll": sem + rng.random(),
                      "tok_max": rng.random() + wrong[i]})
        m3.append({"qid": q, "label_m3": ("FA" if lab == "correct" else ("AH" if i % 4 == 0 else "UH"))
                   if lab != "abstain" else "abstain"})
    d = 12
    tbg = rng.standard_normal((n, 3, d)).astype(np.float16)
    tbg[:, 2, 0] += (3 * wrong).astype(np.float16)              # layer 8 knows correctness / entropy
    slt = rng.standard_normal((n, 3, d)).astype(np.float16)
    subj = rng.standard_normal((n, 3, d)).astype(np.float16)
    subj[:, 1, 0] += (2 * (np.log1p(s_pop) > np.median(np.log1p(s_pop)))).astype(np.float16)
    subj[5] = np.nan                                            # subject not found
    np.save(run / "generation" / "hidden_tbg.npy", tbg)
    np.save(run / "generation" / "hidden_slt.npy", slt)
    (run / "generation" / "layers.json").write_text(json.dumps({"hidden_state_indices": layers}))
    np.save(run / "subject_states" / "hidden_subject.npy", subj)
    for name, rows in (("generation/generations.jsonl", gens), ("correctness/labels.jsonl", labels),
                       ("entropy/features.jsonl", feats), ("subject_states/index.jsonl", gens)):
        (run / name).write_text("\n".join(json.dumps(r) for r in rows))
    if with_m3:
        (run / "labeling" / "labels_m3.jsonl").write_text("\n".join(json.dumps(r) for r in m3))
    return {"run_name": "r1", "output_root": str(tmp_path / "out"), "seed": 0, "milestone": "M4",
            "model": {"id": "Qwen/Qwen2.5-3B-Instruct"}, "data": {"dataset": "popqa"},
            "probes": {"C_grid": [0.01, 0.1], "n_boot": 50, "min_per_class": 10}}


def test_stage_end_to_end(tmp_path):
    from hallu_hierarchy.probes.stage import run
    from hallu_hierarchy.reporting.report import Report

    cfg = _fixture(tmp_path)
    rep = Report(cfg, ["probes"])
    part = rep.stage("probes")
    run(cfg, part)
    rep.finish_stage(part)
    rep.write()
    out = tmp_path / "out" / "r1" / "probes"
    s = json.loads((out / "summary.json").read_text())
    assert set(s["probes"]) == {"sep_sem_entropy", "sep_seq_nll", "correctness", "familiarity"}
    assert s["probes"]["correctness"]["target_auroc"] > 0.85
    assert set(s["probes"]["correctness"]["selected"].values()) == {"tbg@8"}
    assert set(s["probes"]["familiarity"]["selected"].values()) == {"subj@4"}
    assert all(not k.endswith("@0") for k in s["probes"]["correctness"]["by_source"])   # embeddings skipped
    assert s["evals"]["probe_correctness"]["FA_vs_UH"]["pooled"] > 0.8
    rows = [json.loads(l) for l in (out / "oof_scores.jsonl").read_text().splitlines()]
    assert len(rows) == 300 and rows[5]["probe_familiarity"] is None                  # subject missing
    assert rows[7]["probe_correctness"] is None and rows[7]["probe_sep_sem_entropy"] is not None  # abstain
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "hallucination detection" in text and "❌" not in text


def test_stage_without_m3_or_entropy_warns(tmp_path):
    from hallu_hierarchy.probes.stage import run
    from hallu_hierarchy.reporting.report import Report

    cfg = _fixture(tmp_path, with_m3=False)
    (tmp_path / "out" / "r1" / "entropy" / "features.jsonl").unlink()
    rep = Report(cfg, ["probes"])
    part = rep.stage("probes")
    run(cfg, part)
    assert any(c.state == "fail" and "entropy" in c.msg for c in part.checks)
    assert any(c.state == "warn" and "labels_m3" in c.msg for c in part.checks)
    assert "sep_sem_entropy" not in part.summary["probes"] and "correctness" in part.summary["probes"]


def test_hidden_states_found_in_another_input(tmp_path):
    """Kaggle layout: the M3 dataset holds generations.jsonl without hidden states; the generation
    dataset holds them (with its own generations.jsonl) and is searched later."""
    import shutil

    from hallu_hierarchy.probes.stage import load_inputs

    cfg = _fixture(tmp_path)
    run = tmp_path / "out" / "r1"
    other = tmp_path / "gen_ds" / "r1" / "generation"
    other.mkdir(parents=True)
    for f in ("hidden_tbg.npy", "hidden_slt.npy", "layers.json"):
        shutil.move(str(run / "generation" / f), other / f)
    shutil.copy(run / "generation" / "generations.jsonl", other / "generations.jsonl")
    cfg["input_run_dir"] = str(tmp_path / "gen_ds")
    inp = load_inputs(cfg)
    assert set(inp["hidden"]) == {"tbg", "slt", "subject"} and inp["hidden_order_ok"] and inp["layers"] == [0, 4, 8]

    lines = (other / "generations.jsonl").read_text().splitlines()
    (other / "generations.jsonl").write_text("\n".join(lines[1:] + lines[:1]))   # different row order
    assert not load_inputs(cfg)["hidden_order_ok"]
