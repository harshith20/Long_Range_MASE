"""CPU-only tests for M1.5 correctness: strict/relaxed, abstain, flags, overrides, reporting."""

from __future__ import annotations

import csv
import json

import pytest

from hallu_hierarchy.correctness import stage
from hallu_hierarchy.correctness.matching import RefusalPatterns, auto_label, match, relaxed_label
from hallu_hierarchy.correctness.review import ambiguous_subject_heuristic, load_overrides
from hallu_hierarchy.reporting.console import bar, pct
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


def test_overrides_validation(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("qid,decision,gold_noise,ambiguous_subject,note\nq1,Correct,,,ok\nq2,,1,0,\nq3,,,,\n")
    assert load_overrides(p) == {
        "q1": {"decision": "correct", "note": "ok"},
        "q2": {"gold_noise": True, "ambiguous_subject": False},
    }
    assert load_overrides(tmp_path / "missing.csv") == {}
    p.write_text("qid,decision\nq1,maybe\n")
    with pytest.raises(ValueError):
        load_overrides(p)


def test_bar_pct():
    assert pct(34, 200) == "  34 ( 17.0%)"
    assert bar(1.0, 4) == "████" and bar(0.0) == ""


def _gen(qid, subject, pred, aliases, samples):
    return {
        "qid": qid, "split": "train", "question": f"Q about {subject}?", "subject": subject,
        "aliases": aliases, "greedy": {"text": pred}, "samples": [{"text": s} for s in samples],
    }


def test_stage_end_to_end(tmp_path, capsys):
    """Saved M1 run in an input dir -> correctness reads it, compares with M1 labels, applies overrides."""
    src = tmp_path / "kaggle_input" / "outputs" / "r1"
    gens = [
        _gen("q1", "George Rankin", "politician", ["politician"], ["politician", "Unknown"]),
        _gen("q2", "Emma", "Jane Austen", ["Jane Austen"], ["Austen", "Jane Austen"]),
        _gen("q3", "Some Film", "Unknown", ["drama"], ["drama", "comedy"]),
        _gen("q4", "Some Place", "Munich", ["Munich Bavaria"], ["Ulm", "Munich"]),
        _gen("q5", "Atlanta City", "Georgia", ["Fulton County"], ["Georgia", "Georgia"]),
    ]
    (src / "generation").mkdir(parents=True)
    (src / "data").mkdir()
    (src / "correctness").mkdir()
    (src / "generation" / "generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens))
    (src / "data" / "questions.jsonl").write_text(
        "\n".join(json.dumps({"qid": g["qid"], "meta": {"prop": "occupation" if i < 2 else "capital_of"}})
                  for i, g in enumerate(gens))
    )
    # M1 labels had only a `correct` bool
    (src / "correctness" / "labels.jsonl").write_text(
        "\n".join(json.dumps({"qid": g["qid"], "correct": g["qid"] in ("q1", "q2")}) for g in gens)
    )
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("qid,decision,gold_noise\nq4,correct,\nq5,,1\nq_other,wrong,\n")

    cfg = {
        "run_name": "r1", "output_root": str(tmp_path / "out"), "seed": 0,
        "input_run_dir": str(tmp_path / "kaggle_input"),
        "model": {"id": "Qwen/Qwen2.5-3B-Instruct"}, "data": {"dataset": "popqa"},
        "correctness": {"overrides": str(overrides)},
    }
    stage.run(cfg)
    out = tmp_path / "out" / "r1" / "correctness"
    labels = {r["qid"]: r for r in read_jsonl(out / "labels.jsonl")}

    assert labels["q1"]["label"] == "correct" and labels["q1"]["sample_labels"] == ["correct", "abstain"]
    assert labels["q1"]["abstain_frac"] == 0.5
    assert labels["q2"]["flags"]["ambiguous_subject"] and labels["q2"]["exclude_from_training"]
    assert labels["q3"]["label"] == "abstain" and labels["q3"]["label_prev"] == "wrong"
    assert labels["q4"]["label_auto"] == "wrong" and labels["q4"]["label"] == "correct"
    assert labels["q4"]["label_overridden"]
    assert labels["q5"]["flags"]["gold_noise"] and labels["q5"]["label"] == "wrong"

    with open(out / "review" / "disagreements.csv", newline="") as f:
        assert [r["qid"] for r in csv.DictReader(f)] == ["q4"]
    with open(out / "abstain_log.csv", newline="") as f:
        assert {(r["qid"], r["which"]) for r in csv.DictReader(f)} == {("q1", "sample_1"), ("q3", "greedy")}

    s = json.loads((out / "summary.json").read_text())
    assert s["counts"]["strict"] == {"correct": 2, "wrong": 2, "abstain": 1}
    assert s["counts"]["relaxed"] == {"correct": 3, "wrong": 1, "abstain": 1}
    assert s["counts"]["final"] == {"correct": 3, "wrong": 1, "abstain": 1}
    assert s["previous_run"]["changes"] == {"wrong → abstain": 1, "wrong → correct": 1}
    assert s["overrides"] == {"label_changed": 1, "unknown_qids": ["q_other"]}
    assert (out / "meta.json").exists() and (out / "review" / "meta.json").exists()

    printed = capsys.readouterr().out
    assert "STAGE: correctness | qwen2.5-3b | popqa | n=5" in printed
    assert "1 wrong → abstain" in printed

    # rerun: previous labels are now this run's own output -> no changes
    stage.run(cfg)
    assert json.loads((out / "summary.json").read_text())["previous_run"]["changes"] == {}
