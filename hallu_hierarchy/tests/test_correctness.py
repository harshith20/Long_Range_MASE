"""CPU-only tests for correctness (M1.6): strict/relaxed/partial, head words, abstain, flags,
overrides, relation groups, report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hallu_hierarchy.correctness import stage
from hallu_hierarchy.correctness.matching import (
    RefusalPatterns, auto_label, load_head_words, match, strip_head_words,
)
from hallu_hierarchy.correctness.review import (
    ambiguous_subject_heuristic, clean_aliases, load_generic_aliases, load_overrides,
    load_relation_groups,
)
from hallu_hierarchy.reporting.console import bar, pct
from hallu_hierarchy.reporting.report import Report
from hallu_hierarchy.run import main, parse_stages
from hallu_hierarchy.utils import PKG_DIR, read_jsonl

CFG = PKG_DIR / "configs"
REFUSAL = RefusalPatterns.load(CFG / "refusal_patterns.yaml")
HEAD = load_head_words(CFG / "head_words.yaml")


def label_of(pred, aliases, relation=None):
    return auto_label(match(pred, aliases, head_words=HEAD), REFUSAL, relation)[0]


@pytest.mark.parametrize(
    "pred,aliases,strict,relaxed",
    [
        ("Munich", ["Munich", "München"], True, True),
        ("Munich, Germany", ["Munich"], True, True),      # alias inside prediction
        ("Ulm", ["Munich"], False, False),                # never similarity-based
        ("Munichville", ["Munich"], False, False),        # whole words only
        ("Munich", ["Munich Bavaria"], False, True),      # prediction inside alias, >= 6 chars
        ("Drama", ["legal drama"], False, False),         # too short for relaxed
        ("John Smith", ["Sir John Smith Jr"], False, True),  # 2 words
        ("", ["Munich"], False, False),
        ("politician", ["politician", ""], True, True),   # empty alias ignored
    ],
)
def test_strict_relaxed(pred, aliases, strict, relaxed):
    m = match(pred, aliases)
    assert m["match_strict"] is strict
    assert m["match_relaxed"] is relaxed


def test_head_word_stripping():
    assert HEAD[0] in ("tv series", "video game")          # multi-word head words tried first
    assert strip_head_words("comedy film", HEAD) == "comedy"
    assert strip_head_words("science fiction film", HEAD) == "science fiction"
    assert strip_head_words("drama tv series", HEAD) == "drama"
    assert strip_head_words("film", HEAD) is None          # never stripped to nothing
    assert strip_head_words("munich", HEAD) is None        # nothing to strip
    m = match("Comedy", ["comedy film"], head_words=HEAD)
    assert m["match_strict"] and m["strict_via_head_word"]
    assert not match("Comedy", ["comedy film"])["match_strict"]  # needs the stripped copy


@pytest.mark.parametrize(
    "pred,aliases,relation,label",
    [
        ("Comedy", ["comedy film"], "genre", "correct"),               # head word stripped
        ("Fiction", ["science fiction film"], "genre", "partial"),     # less specific
        ("Singer", ["singer-songwriter"], "occupation", "partial"),
        ("New York City", ["Queens, New York City"], "place of birth", "partial"),
        ("Drama", ["legal drama"], "genre", "wrong"),                   # too short for partial
        ("Ulm", ["Munich"], "place of birth", "wrong"),
        ("Unknown", ["Unknown"], "genre", "abstain"),                   # refusal wins
        ("None", ["Islam"], "occupation", "abstain"),
        ("None", ["Islam"], "religion", "wrong"),                       # religion exemption
        ("None", ["irreligion", "none"], "religion", "correct"),
        ("Catholic", ["Catholic Church"], "religion", "correct"),      # head word "church"
        ("Muslim", ["Muslim religion"], "religion", "correct"),        # head word "religion"
        ("Anglican Church", ["Catholic Church"], "religion", "wrong"),  # stripping never loosens
        ("Christianity", ["Protestant Christianity"], "religion", "partial"),
        ("Mennonite", ["Mennonite Church USA"], "religion", "partial"),  # not a trailing head word
    ],
)
def test_labels(pred, aliases, relation, label):
    assert label_of(pred, aliases, relation) == label


@pytest.mark.parametrize(
    "pred,pattern",
    [
        ("Unknown.", "unknown"),
        ("None", "none"),
        ("NoneGiven", "nonegiven"),
        ("None given", "none given"),
        ("Various", "various"),
        ("Not enough information", "not enough information"),
        ("Pygmalion is not a person", "is not"),
        ("", "<empty>"),
        ("Not specified.", "not specified"),             # E06
        ("Unspecified", "unspecified"),
        ("Not available", "not available"),
        ("Not specified in Hinduism", None),              # exact only: a longer answer stays an answer
        ("Unknown Pleasures", None),                      # exact patterns need the whole answer
        ("Munich", None),
    ],
)
def test_refusal(pred, pattern):
    assert REFUSAL.find(match(pred, ["x"])["pred_norm"]) == pattern


def test_ambiguous_heuristic():
    cfg = {"ambiguous_single_word": True, "ambiguous_subjects": ["The Messiah Project"]}
    assert ambiguous_subject_heuristic("Emma", cfg)
    assert ambiguous_subject_heuristic("the Messiah project", cfg)
    assert not ambiguous_subject_heuristic("George Rankin", cfg)
    assert not ambiguous_subject_heuristic("Emma", {"ambiguous_single_word": False})


def test_generic_alias_cleaning():
    generic = load_generic_aliases(CFG / "generic_aliases.yaml")
    assert {"church", "film", "music", "movie", "series"} <= generic
    assert clean_aliases(["Catholic Church", "Church"], generic) == (["Catholic Church"], ["Church"])
    assert clean_aliases(["Film"], generic) == (["Film"], [])  # never drop every alias
    kept, _ = clean_aliases(["Catholic Church", "Church"], generic)
    assert not match("Anglican Church", kept)["match_strict"]


def test_relation_groups_config():
    groups, borderline = load_relation_groups(CFG / "relation_groups.yaml")
    assert groups["capital of"] == "closed" and groups["screenwriter"] == "open"
    assert groups["occupation"] == "open" and borderline == {"occupation"}
    assert len([g for g in groups.values() if g == "closed"]) == 7


def test_committed_overrides_file():
    ov = load_overrides(PKG_DIR / "overrides" / "popqa_qwen2.5-3b.csv")
    assert ov["popqa-1919197"]["label"] == "correct"
    assert ov["popqa-1823075"]["label"] == "wrong" and ov["popqa-1823075"]["gold_noise"] is True
    assert ov["popqa-384830"]["label"] == "correct" and not ov["popqa-384830"]["ambiguous_subject"]


def test_overrides_validation(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("qid,label,gold_noise,ambiguous_subject,note\nq1,Partial,,,ok\nq2,,1,0,\nq3,,,,\n")
    assert load_overrides(p) == {
        "q1": {"label": "partial", "note": "ok"},
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
        parse_stages(["eval"])  # not a stage (yet)


def _gen(qid, subject, pred, aliases, samples):
    return {
        "qid": qid, "split": "train", "question": f"Q about {subject}?", "subject": subject,
        "aliases": aliases, "greedy": {"text": pred}, "samples": [{"text": s} for s in samples],
    }


def _saved_run(tmp_path):
    """A saved earlier run (M1.5 labels) inside a fake Kaggle input folder."""
    src = tmp_path / "kaggle_input" / "outputs" / "r1"
    rows = [  # (gen, relation, previous label)
        (_gen("q1", "George Rankin", "politician", ["politician"], ["politician", "Unknown"]), "occupation", "correct"),
        (_gen("q2", "Emma", "Comedy", ["comedy film"], ["Comedy", "Drama"]), "genre", "wrong"),
        (_gen("q3", "Some Film", "Unknown", ["drama"], ["drama", "comedy"]), "genre", "abstain"),
        (_gen("q4", "Some Place", "Munich", ["Munich Bavaria"], ["Ulm", "Munich"]), "capital", "wrong"),
        (_gen("q5", "Atlanta City", "Georgia", ["Fulton County"], ["Georgia", "Georgia"]), "capital of", "wrong"),
        (_gen("q6", "Ann Lee", "None", ["Islam"], ["None", "Islam"]), "religion", "abstain"),
        (_gen("q7", "Bob Jones", "Singer", ["singer-songwriter"], ["Singer", "Actor"]), "occupation", "wrong"),
        (_gen("q8", "Kim Ray", "Ulm", ["Munich"], ["Ulm", "Ulm"]), "place of birth", "wrong"),
    ]
    for d in ("generation", "data", "correctness"):
        (src / d).mkdir(parents=True)
    (src / "generation" / "generations.jsonl").write_text("\n".join(json.dumps(g) for g, _, _ in rows))
    (src / "data" / "questions.jsonl").write_text(
        "\n".join(json.dumps({"qid": g["qid"], "meta": {"prop": rel}}) for g, rel, _ in rows))
    (src / "correctness" / "labels.jsonl").write_text(
        "\n".join(json.dumps({"qid": g["qid"], "label": prev}) for g, _, prev in rows))


def _cfg(tmp_path, overrides):
    return {
        "run_name": "r1", "output_root": str(tmp_path / "out"), "seed": 0, "milestone": "M1.6",
        "input_run_dir": str(tmp_path / "kaggle_input"),
        "model": {"id": "Qwen/Qwen2.5-3B-Instruct"}, "data": {"dataset": "popqa"},
        "correctness": {"overrides": str(overrides), "expected_accuracy": [0.0, 1.0], "max_partial_rate": 0.5,
                        "max_relation_abstain_rate": 1.0},
    }


def _run(cfg):
    report = Report(cfg, ["correctness"])
    part = report.stage("correctness")
    stage.run(cfg, part)
    report.finish_stage(part)
    report.write()
    return report


def test_stage_end_to_end(tmp_path):
    _saved_run(tmp_path)
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("qid,label,gold_noise,ambiguous_subject,note\nq4,correct,,,\nq5,,1,,\nq_other,wrong,,,\n")
    cfg = _cfg(tmp_path, overrides)
    report = _run(cfg)

    out = tmp_path / "out" / "r1" / "correctness"
    labels = {r["qid"]: r for r in read_jsonl(out / "labels.jsonl")}
    assert labels["q1"]["label"] == "correct" and labels["q1"]["sample_labels"] == ["correct", "abstain"]
    assert labels["q1"]["relation_group"] == "open" and labels["q1"]["dataset"] == "popqa"
    assert labels["q2"]["label"] == "correct" and labels["q2"]["strict_via_head_word"]
    assert labels["q2"]["flags"]["ambiguous_subject"] and not labels["q2"]["exclude_from_training"]
    assert labels["q3"]["label"] == "abstain" and labels["q3"]["exclude_from_training"]
    assert labels["q4"]["label_auto"] == "partial" and labels["q4"]["label"] == "correct"
    assert labels["q5"]["flags"]["gold_noise"] and labels["q5"]["exclude_from_training"]
    assert labels["q6"]["label"] == "wrong" and labels["q6"]["refusal_pattern"] is None  # religion "None"
    assert labels["q7"]["label"] == "partial" and labels["q7"]["exclude_from_training"]
    assert labels["q8"]["label"] == "wrong" and not labels["q8"]["exclude_from_training"]

    s = json.loads((out / "summary.json").read_text())
    assert s["counts_auto"] == {"correct": 2, "wrong": 3, "abstain": 1, "partial": 2}
    assert s["counts"] == {"correct": 3, "wrong": 3, "abstain": 1, "partial": 1}
    assert s["previous_run"]["changes"] == {
        "wrong → correct": 1, "wrong → correct (overrides)": 1, "abstain → wrong": 1, "wrong → partial": 1}
    assert s["head_word_matches"] == 1
    assert s["by_group"]["closed"]["n"] == 5 and s["by_group"]["open"]["n"] == 3
    assert s["borderline"]["occupation"]["n"] == 2
    assert s["overrides"]["unknown_qids"] == ["q_other"] and s["overrides"]["rows_applied"] == 2
    assert s["excluded_from_training"] == 3  # q3 abstain, q5 gold_noise, q7 partial
    assert [e["qid"] for e in s["partial_examples"]] == ["q7"]
    assert (out / "partial.csv").exists() and (tmp_path / "out" / "r1" / "summary.json").exists()

    text = (tmp_path / "out" / "REPORT.md").read_text()
    assert report.status == "pass", text
    assert "Milestone: M1.6 | qwen2.5-3b | popqa | n=8 | stages: correctness" in text
    assert "ACTION NEEDED" not in text and not (tmp_path / "out" / "REVIEW.csv").exists()
    assert "partial 1 (12.5%)" in text and "head-word matches: 1" in text
    assert "(occupation: borderline)" in text
    assert len(text.split("FOR CLAUDE")[1].strip().splitlines()) <= 62


def test_partial_rate_check_warns(tmp_path):
    _saved_run(tmp_path)
    cfg = _cfg(tmp_path, tmp_path / "none.csv")
    cfg["correctness"]["max_partial_rate"] = 0.05
    report = _run(cfg)
    assert report.status == "warn"
    assert "partial rate 25.0% (expected < 5%" in (tmp_path / "out" / "REPORT.md").read_text()


def test_crash_still_writes_report(tmp_path):
    """A failing stage gives STATUS FAIL with the error, exit code 1."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        f"run_name: nothing_here\nseed: 0\nmilestone: M1.6\noutput_root: {tmp_path / 'out'}\n"
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
    _, cut = tbg_attention_per_head(att, [], q_pos=4, row_layers=[0], max_len=3)
    assert cut


def test_spop_deciles_and_test_counts():
    from hallu_hierarchy.correctness.stage import _eval_counts, spop_bins

    rows = [{"s_pop": i, "label": "correct" if i >= 15 else "wrong", "relation_group": "closed" if i % 2 else "open",
             "split": "test" if i % 4 == 0 else "train", "exclude_from_training": False} for i in range(20)]
    bins = spop_bins(rows, k=4)
    assert [b["n"] for b in bins] == [5, 5, 5, 5]
    assert bins[0]["s_pop_min"] == 0 and bins[-1]["s_pop_max"] == 19
    assert bins[0]["acc"] == 0.0 and bins[-1]["acc"] == 1.0
    assert bins[-1]["n_closed"] + bins[-1]["n_open"] == 5
    assert spop_bins([{"s_pop": None}], k=4) == []
    ec = _eval_counts(rows, min_test=5)          # CV: all eligible rows count (15 wrong, 5 correct)
    assert ec == {"eval_correct": 5, "eval_wrong": 15, "usable": True}
    rows[19]["exclude_from_training"] = True
    assert _eval_counts(rows, min_test=5)["usable"] is False


def test_report_relation_checks(tmp_path):
    """High-abstain relations WARN and show examples; usable-relation and s_pop lines appear."""
    _saved_run(tmp_path)
    cfg = _cfg(tmp_path, tmp_path / "none.csv")
    cfg["correctness"].update({"max_relation_abstain_rate": 0.4, "within_relation_min_test": 1})
    report = _run(cfg)
    text = (tmp_path / "out" / "REPORT.md").read_text()
    assert report.status == "warn"
    assert "high abstain in genre 50%" in text           # q3 of 2 genre rows abstains
    assert "abstain audit" in text and "[genre] q3 'Unknown' → unknown" in text
    assert "within-relation usable: 0 of" in text        # fixture has no test split
    s = json.loads((tmp_path / "out" / "r1" / "correctness" / "summary.json").read_text())
    assert s["by_relation"][0]["top_refusal"] in ("-", "unknown ×1")


def test_subject_folds():
    from hallu_hierarchy.data.splits import assign_subject_folds, check_folds

    items = [{"subject_id": f"s{i % 23}"} for i in range(100)]
    f = assign_subject_folds(items, 5, seed=0)
    assert set(f.values()) == {0, 1, 2, 3, 4} and len(f) == 23
    sizes = [list(f.values()).count(k) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1                      # subjects dealt round-robin
    assert f == assign_subject_folds(items, 5, seed=0)       # deterministic
    assert f != assign_subject_folds(items, 5, seed=1)
    check_folds([{"subject_id": it["subject_id"], "fold": f[it["subject_id"]]} for it in items])
    with pytest.raises(AssertionError):
        check_folds([{"subject_id": "a", "fold": 0}, {"subject_id": "a", "fold": 1}])
    with pytest.raises(ValueError):
        assign_subject_folds(items, 1, seed=0)


def test_stage_writes_folds(tmp_path):
    _saved_run(tmp_path)
    cfg = _cfg(tmp_path, tmp_path / "none.csv")
    cfg["evaluation"] = {"folds": 3}
    _run(cfg)
    labels = read_jsonl(tmp_path / "out" / "r1" / "correctness" / "labels.jsonl")
    assert {r["fold"] for r in labels} <= {0, 1, 2} and all(r["subject_id"] for r in labels)
    text = (tmp_path / "out" / "REPORT.md").read_text()
    assert "✅ 3-fold CV grouped by subject: no subject in two folds" in text
    assert "eval c/w" in text


def test_discover_subject_states_and_label_precedence(tmp_path, monkeypatch):
    """Mirrors the real Kaggle inputs: generation outputs + old labels in one dataset,
    subject states alone in another. The subject-states dataset must be found, and old labels
    are only picked when the run itself has not produced labels."""
    from hallu_hierarchy import utils

    run = "qwen2.5-3b_popqa_full"
    old = tmp_path / "input" / "hallu-hierarchy-m-1-5" / "outputs" / run
    ss = tmp_path / "input" / "hallu-hierarchy-subject-states" / "outputs" / run
    for d in (old / "generation", old / "correctness", ss / "subject_states"):
        d.mkdir(parents=True)
    (old / "correctness" / "labels.jsonl").write_text("old")
    assert utils.discover_kaggle_inputs(run, root=tmp_path / "input") == [old, ss]

    monkeypatch.setattr(utils, "KAGGLE_INPUT", tmp_path / "input")
    cfg = {"run_name": run, "output_root": str(tmp_path / "working")}
    assert utils.find_upstream(cfg, "subject_states/hidden_subject.npy") is None  # file absent in fixture
    (ss / "subject_states" / "hidden_subject.npy").write_text("x")
    assert utils.find_upstream(cfg, "subject_states/hidden_subject.npy") == ss / "subject_states" / "hidden_subject.npy"
    # without a correctness stage in this run, the OLD labels from the dataset are picked
    assert utils.find_upstream(cfg, "correctness/labels.jsonl") == old / "correctness" / "labels.jsonl"
    # running correctness first in the same run puts fresh labels in the run folder, which wins
    fresh = tmp_path / "working" / run / "correctness" / "labels.jsonl"
    fresh.parent.mkdir(parents=True)
    fresh.write_text("new")
    assert utils.find_upstream(cfg, "correctness/labels.jsonl") == fresh
