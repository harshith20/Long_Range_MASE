"""CPU-only tests for M1.5 correctness: strict/relaxed, abstain, flags, overrides, reporting."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from hallu_hierarchy.correctness import stage
from hallu_hierarchy.correctness.matching import RefusalPatterns, auto_label, match, relaxed_label
from hallu_hierarchy.correctness.review import (
    ambiguous_subject_heuristic, clean_aliases, load_generic_aliases, load_overrides,
)
from hallu_hierarchy.reporting.console import bar, pct
from hallu_hierarchy.reporting.report import Report
from hallu_hierarchy.run import main, parse_stages
from hallu_hierarchy.utils import PKG_DIR, read_jsonl

REFUSAL = RefusalPatterns.load(PKG_DIR / "configs" / "refusal_patterns.yaml")


@pytest.mark.parametrize(
    "pred,aliases,strict,relaxed",
    [
        ("Munich", ["Munich", "München"], True, True),
        ("München.", ["Munich"], False, False),          # different alias, accents folded only
        ("Munich, Germany", ["Munich"], True, True),      # alias inside prediction
        ("Ulm", ["Munich"], False, False),                # never similarity-based
        ("Munichville", ["Munich"], False, False),        # whole words only
        ("Munich", ["Munich Bavaria"], False, True),      # prediction inside alias, >= 6 chars
        ("Drama", ["legal drama"], False, False),         # too short for relaxed
        ("Legal", ["legal drama"], False, False),         # 5 chars, 1 word
        ("John Smith", ["Sir John Smith Jr"], False, True),  # 2 words
        ("", ["Munich"], False, False),
        ("politician", ["politician", ""], True, True),   # empty alias ignored
    ],
)
def test_strict_relaxed(pred, aliases, strict, relaxed):
    m = match(pred, aliases)
    assert m["match_strict"] is strict
    assert m["match_relaxed"] is relaxed


@pytest.mark.parametrize(
    "pred,pattern",
    [
        ("Unknown.", "unknown"),
        ("None", "none"),
        ("NoneGiven", "nonegiven"),
        ("Various", "various"),
        ("Not enough information", "not enough information"),
        ("Pygmalion is not a person", "is not"),          # article stripped from pattern
        ("", "<empty>"),
        ("Unknown Pleasures", None),                      # exact patterns need the whole answer
        ("Munich", None),
    ],
)
def test_refusal(pred, pattern):
    assert REFUSAL.find(match(pred, ["x"])["pred_norm"]) == pattern


def test_relaxed_never_changes_label():
    m = match("Munich", ["Munich Bavaria"])
    assert auto_label(m, REFUSAL) == ("wrong", None)
    assert relaxed_label(m, "wrong") == "correct"
    m = match("Unknown", ["Unknown"])  # refusal wins even if it equals an alias
    assert auto_label(m, REFUSAL)[0] == "abstain" and relaxed_label(m, "abstain") == "abstain"


def test_ambiguous_heuristic():
    cfg = {"ambiguous_single_word": True, "ambiguous_subjects": ["The Messiah Project"]}
    assert ambiguous_subject_heuristic("Emma", cfg)
    assert ambiguous_subject_heuristic("the Messiah project", cfg)
    assert not ambiguous_subject_heuristic("George Rankin", cfg)
    assert not ambiguous_subject_heuristic("Emma", {"ambiguous_single_word": False})


def test_generic_alias_cleaning():
    generic = load_generic_aliases(PKG_DIR / "configs" / "generic_aliases.yaml")
    assert {"church", "film", "music", "movie", "series"} <= generic
    assert clean_aliases(["Catholic Church", "Church"], generic) == (["Catholic Church"], ["Church"])
    assert clean_aliases(["Film"], generic) == (["Film"], [])  # never drop every alias
    # "Anglican Church" no longer matches gold "Catholic Church" through the alias "Church"
    kept, _ = clean_aliases(["Catholic Church", "Church"], generic)
    assert not match("Anglican Church", kept)["match_strict"]
    assert match("Anglican Church", ["Catholic Church", "Church"])["match_strict"]


def test_committed_overrides_file():
    ov = load_overrides(PKG_DIR / "overrides" / "popqa_qwen2.5-3b.csv")
    assert ov["popqa-1919197"]["label"] == "correct"
    assert ov["popqa-1823075"] == {**ov["popqa-1823075"], "label": "wrong", "gold_noise": True}
    assert ov["popqa-384830"]["label"] == "correct" and not ov["popqa-384830"]["ambiguous_subject"]


def test_overrides_validation(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("qid,label,gold_noise,ambiguous_subject,note\nq1,Correct,,,ok\nq2,,1,0,\nq3,,,,\n")
    assert load_overrides(p) == {
        "q1": {"label": "correct", "note": "ok"},
        "q2": {"gold_noise": True, "ambiguous_subject": False},
        "q3": {},
    }
    assert load_overrides(tmp_path / "missing.csv") == {}
    p.write_text("qid,label\nq1,maybe\n")
    with pytest.raises(ValueError):
        load_overrides(p)


def test_bar_pct():
    assert pct(34, 200) == "34 (17.0%)"
    assert bar(1.0, 4) == "████" and bar(0.0) == ""


def test_parse_stages():
    assert parse_stages(["generation,correctness"]) == ["generation", "correctness"]
    assert parse_stages(["data", "generate"]) == ["data", "generation"]
    assert parse_stages(None) is None
    with pytest.raises(SystemExit):
        parse_stages(["entropy"])


def _gen(qid, subject, pred, aliases, samples):
    return {
        "qid": qid, "split": "train", "question": f"Q about {subject}?", "subject": subject,
        "aliases": aliases, "greedy": {"text": pred}, "samples": [{"text": s} for s in samples],
    }


def _saved_m1_run(tmp_path):
    """A saved M1 run inside a fake Kaggle input folder."""
    src = tmp_path / "kaggle_input" / "outputs" / "r1"
    gens = [
        _gen("q1", "George Rankin", "politician", ["politician"], ["politician", "Unknown"]),
        _gen("q2", "Emma", "Jane Austen", ["Jane Austen"], ["Austen", "Jane Austen"]),
        _gen("q3", "Some Film", "Unknown", ["drama"], ["drama", "comedy"]),
        _gen("q4", "Some Place", "Munich", ["Munich Bavaria"], ["Ulm", "Munich"]),
        _gen("q5", "Atlanta City", "Georgia", ["Fulton County"], ["Georgia", "Georgia"]),
        _gen("q6", "Edmund Walsh", "Anglican Church", ["Catholic Church", "Church"], ["Catholic", "Church"]),
        _gen("q7", "Bob Jones", "Paris France", ["Paris"], ["Paris", "Lyon"]),
        _gen("q8", "Ann Lee", "Lyon Rhone", ["Lyon Rhone Alpes"], ["Lyon", "Lyon"]),
    ]
    for d in ("generation", "data", "correctness"):
        (src / d).mkdir(parents=True)
    (src / "generation" / "generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens))
    (src / "data" / "questions.jsonl").write_text(
        "\n".join(json.dumps({"qid": g["qid"], "meta": {"prop": "occupation" if i < 2 else "capital_of"}})
                  for i, g in enumerate(gens))
    )
    # M1 labels had only a `correct` bool (q6 was wrongly correct through the alias "Church")
    (src / "correctness" / "labels.jsonl").write_text(
        "\n".join(json.dumps({"qid": g["qid"], "correct": g["qid"] in ("q1", "q2", "q6", "q7")}) for g in gens)
    )
    return gens


def _cfg(tmp_path, overrides):
    return {
        "run_name": "r1", "output_root": str(tmp_path / "out"), "seed": 0, "milestone": "M1.5",
        "input_run_dir": str(tmp_path / "kaggle_input"),
        "model": {"id": "Qwen/Qwen2.5-3B-Instruct"}, "data": {"dataset": "popqa"},
        "correctness": {"overrides": str(overrides), "max_ambiguous_frac": 0.5, "expected_accuracy": [0.0, 1.0]},
    }


def test_stage_end_to_end(tmp_path):
    _saved_m1_run(tmp_path)
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("qid,label,gold_noise,ambiguous_subject,note\nq4,correct,,,\nq5,,1,,\nq_other,wrong,,,\n")
    cfg = _cfg(tmp_path, overrides)

    report = Report(cfg, ["correctness"])
    part = report.stage("correctness")
    stage.run(cfg, part)
    report.finish_stage(part)
    report.write()

    out = tmp_path / "out" / "r1" / "correctness"
    labels = {r["qid"]: r for r in read_jsonl(out / "labels.jsonl")}
    assert labels["q1"]["label"] == "correct" and labels["q1"]["sample_labels"] == ["correct", "abstain"]
    assert labels["q1"]["abstain_frac"] == 0.5 and labels["q1"]["dataset"] == "popqa"
    assert labels["q2"]["flags"]["ambiguous_subject"]
    assert not labels["q2"]["exclude_from_training"]          # ambiguous but correct: kept
    assert labels["q3"]["label"] == "abstain" and labels["q3"]["label_prev"] == "wrong"
    assert labels["q4"]["label_auto"] == "wrong" and labels["q4"]["label"] == "correct"
    assert labels["q5"]["flags"]["gold_noise"] and labels["q5"]["exclude_from_training"]
    assert labels["q6"]["label"] == "wrong" and labels["q6"]["aliases_dropped"] == ["Church"]
    assert labels["q6"]["sample_labels"] == ["wrong", "wrong"]  # "Catholic": relaxed only

    s = json.loads((out / "summary.json").read_text())
    assert s["counts"]["strict"] == {"correct": 3, "wrong": 4, "abstain": 1}
    assert s["counts"]["final"] == {"correct": 4, "wrong": 3, "abstain": 1}
    assert s["previous_run"]["changes"] == {"wrong → abstain": 1, "wrong → correct (overrides)": 1, "correct → wrong": 1}
    assert s["overrides"]["unknown_qids"] == ["q_other"] and s["overrides"]["rows_applied"] == 2
    assert s["review_rows"] == 1                                  # q8 (q4 has an override)
    assert s["aliases_dropped"] == {"church": 1}
    assert (out / "meta.json").exists() and (tmp_path / "out" / "r1" / "summary.json").exists()

    with open(tmp_path / "out" / "REVIEW.csv", newline="") as f:
        assert [r["qid"] for r in csv.DictReader(f)] == ["q8"]
    text = (tmp_path / "out" / "REPORT.md").read_text()
    assert text.startswith("==================== REPORT START")
    assert text.rstrip().endswith("REPORT END ======================")
    assert "STATUS: ⚠️ WARN" in text                           # a review row is open
    assert "Milestone: M1.5 | qwen2.5-3b | popqa | n=8 | stages: correctness" in text
    assert "ACTION NEEDED" in text and "q8 | " in text
    assert "1 wrong → abstain" in text and "aliases dropped as generic: 1 (church ×1)" in text
    block = text.split("FOR CLAUDE")[1]
    assert len(block.strip().splitlines()) <= 62

    # decide q8 -> no review rows, REVIEW.csv removed, previous labels now this run's own
    overrides.write_text(overrides.read_text() + "q8,correct,,,\n")
    report = Report(cfg, ["correctness"])
    part = report.stage("correctness")
    stage.run(cfg, part)
    report.write()
    assert not (tmp_path / "out" / "REVIEW.csv").exists()
    assert "ACTION NEEDED" not in (tmp_path / "out" / "REPORT.md").read_text()
    assert report.status == "pass"
    s = json.loads((out / "summary.json").read_text())
    assert s["previous_run"]["changes"] == {"wrong → correct (overrides)": 1}


def test_crash_still_writes_report(tmp_path):
    """A failing stage gives STATUS FAIL with the error, exit code 1."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"run_name: nothing_here\nseed: 0\nmilestone: M1.5\noutput_root: {tmp_path / 'out'}\n"
        f"input_run_dir: {tmp_path / 'missing'}\nmodel: {{id: Qwen/Qwen2.5-3B-Instruct}}\n"
        "data: {dataset: popqa}\n"
    )
    assert main(["--config", str(cfg_path), "--stages", "correctness"]) == 1
    text = (tmp_path / "out" / "REPORT.md").read_text()
    assert "STATUS: ❌ FAIL" in text and "FileNotFoundError" in text


def test_discover_kaggle_inputs(tmp_path):
    from hallu_hierarchy.utils import discover_kaggle_inputs

    run = tmp_path / "datasets" / "user" / "hallu-m1-exp1" / "outputs" / "qwen2.5-3b_popqa_dev200"
    (run / "generation").mkdir(parents=True)
    (tmp_path / "other" / "qwen2.5-3b_popqa_dev200").mkdir(parents=True)  # no stage folders: ignored
    assert discover_kaggle_inputs("qwen2.5-3b_popqa_dev200", root=tmp_path) == [run]


def test_tbg_attention_per_head():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from hallu_hierarchy.generation.generate import attention_row_layers, tbg_attention_per_head

    assert attention_row_layers(36, [0.5, 0.75, 0.9]) == [18, 26, 32]
    L, H, S = 4, 2, 6
    att = tuple(torch.softmax(torch.randn(1, H, S, S), -1) for _ in range(L))
    arrs, cut = tbg_attention_per_head(att, [1, 2], q_pos=4, row_layers=[2, 3], max_len=8)
    assert arrs["attn_subj_heads_tbg"].shape == (L, H) and not cut
    assert np.allclose(arrs["attn_subj_heads_tbg"][2], att[2][0, :, 4, [1, 2]].sum(-1).numpy(), atol=1e-6)
    rows = arrs["attn_row_tbg"]
    assert rows.shape == (2, H, 8) and np.isnan(rows[:, :, 5:].astype(np.float32)).all()
    _, cut = tbg_attention_per_head(att, [], q_pos=4, row_layers=[0], max_len=3)
    assert cut
