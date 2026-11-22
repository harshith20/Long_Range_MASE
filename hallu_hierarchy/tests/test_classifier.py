"""CPU tests for M5: metric helpers and the classifier stage end to end on synthetic inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from hallu_hierarchy.classifier.stage import auprc, columns, ece


def test_auprc_and_ece():
    y = np.array([1, 1, 0, 0], bool)
    assert auprc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 1.0
    assert auprc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx((1 / 3 + 2 / 4) / 2)
    assert ece(np.array([1.0, 1.0, 0.0, 0.0]), y) == 0.0


def test_feature_sets_are_single_pass():
    full = columns("E+L+A+F+C")
    assert "sem_entropy" not in full and "seq_nll" not in full        # sampled features never at inference
    assert "abstain_frac" not in full and "relation" not in full


def _fixture(tmp_path, n=600):
    run = tmp_path / "out" / "r1"
    for d in ("labeling", "entropy", "probes"):
        (run / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    m3, feats, probes = [], [], []
    for i in range(n):
        q = f"q{i}"
        cls = rng.choice(["FA", "AH", "UH"], p=[0.3, 0.2, 0.5])
        rel = ["capital", "director", "genre"][i % 3]
        hall = cls != "FA"
        fam = 0.7 if cls in ("FA", "AH") else 0.3
        m3.append({"qid": q, "label_m3": str(cls), "fold": i % 5, "relation": rel,
                   "relation_group": "open" if rel == "director" else "closed", "subject_id": str(i),
                   "flags": {"ambiguous_subject": i % 7 == 0}})
        e = 1.0 * hall + rng.normal(0, 0.5)
        feats.append({"qid": q, "fold": i % 5, "tok_mean": e + 1, "tok_max": e + 1.5, "sem_entropy": e,
                      **{k: rng.random() for k in ("lens_H_18", "lens_H_24", "lens_H_30", "lens_H_36")},
                      "lens_lstar": float(rng.integers(20, 37)),
                      **{f"attn_{a}_{l}": (np.nan if i == 3 else rng.random()) for a in ("mean", "max") for l in (18, 26, 32)}})
        probes.append({"qid": q, "fold": i % 5, "probe_sep_sem_entropy": 1 / (1 + np.exp(-2 * e)),
                       "probe_sep_seq_nll": rng.random(), "probe_correctness": 1 / (1 + np.exp(-2 * e)),
                       "probe_familiarity": fam + rng.normal(0, 0.15)})
    for name, rows in (("labeling/labels_m3.jsonl", m3), ("entropy/features.jsonl", feats), ("probes/oof_scores.jsonl", probes)):
        (run / name).write_text("\n".join(json.dumps(r) for r in rows).replace("NaN", "null"))
    (run / "entropy" / "summary.json").write_text(json.dumps({"n": n, "nli": {"seconds": 60.0}}))
    return {"run_name": "r1", "output_root": str(tmp_path / "out"), "seed": 0, "milestone": "M5",
            "model": {"id": "Qwen/Qwen2.5-3B-Instruct"}, "data": {"dataset": "popqa"},
            "classifier": {"n_boot": 30, "min_per_class": 10, "gb": {"max_iter": 30}}}


def test_stage_end_to_end(tmp_path):
    from hallu_hierarchy.classifier.stage import run
    from hallu_hierarchy.reporting.report import Report

    cfg = _fixture(tmp_path)
    rep = Report(cfg, ["classifier"])
    part = rep.stage("classifier")
    run(cfg, part)
    rep.finish_stage(part)
    rep.write()
    s = json.loads((tmp_path / "out" / "r1" / "classifier" / "summary.json").read_text())
    hall = s["tasks"]["hall"]
    assert hall["models"]["lr:E"]["pooled"] > 0.85                         # entropy separates FA vs hallucination
    assert s["tasks"]["fa_ah"]["models"]["lr:E+L+A+F"]["pooled"] > s["tasks"]["fa_ah"]["models"]["lr:E"]["pooled"] - 0.02
    assert s["tasks"]["ah_uh"]["models"]["lr:E+L+A+F"]["pooled"] > 0.8    # familiarity separates AH from UH
    assert abs(hall["baselines"]["relation-only"]["within"] - 0.5) < 0.15   # ~constant within a relation (per-fold refits)
    assert set(s["flat"]) == {"flat", "two_stage"}
    assert s["timing"]["nli_s_per_q"] == pytest.approx(0.1)
    rows = (tmp_path / "out" / "r1" / "classifier" / "oof_predictions.jsonl").read_text().splitlines()
    assert len(rows) == 600
    text = (tmp_path / "out" / "REPORT.md").read_text(encoding="utf-8")
    assert "FA vs AH (H2/H3)" in text and "❌" not in text
