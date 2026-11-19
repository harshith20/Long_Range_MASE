"""CPU tests for M2 entropy features, metrics and the stage (fake NLI, no model download)."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from hallu_hierarchy.entropy.features import (
    attention_features, cluster_by_entailment, content_slice, lens_features, semantic_entropy,
    sequence_nll, token_features, unique_answers,
)
from hallu_hierarchy.eval.metrics import auroc, spearman_matrix, vif_from_corr


# ----------------------------------------------------------------------------- metrics

def test_auroc_matches_brute_force():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=60), 1)              # rounding creates ties
    y = rng.random(60) < 0.4
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0
                     for a, b in itertools.product(s[y], s[~y])])
    assert auroc(s, y) == pytest.approx(brute)
    assert auroc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert auroc(np.array([1, np.nan, 3.0]), np.array([0, 1, 1], bool)) == 1.0   # NaN dropped
    assert auroc(np.array([1.0, 2.0]), np.array([1, 1], bool)) is None          # one class only


def test_spearman_and_vif():
    x = np.arange(20, dtype=float)
    names, c = spearman_matrix({"a": x, "b": x ** 3, "c": -x})
    assert names == ["a", "b", "c"] and np.allclose(c[0, 1], 1) and np.allclose(c[0, 2], -1)
    assert np.allclose(vif_from_corr(np.eye(3)), 1.0)


# ----------------------------------------------------------------------------- features

def test_token_and_sequence_features():
    g = {"token_entropy": [2.0, 1.0, 9.0], "ends_with_eos": True}       # last = EOS step, dropped
    assert token_features(g) == {"tok_mean": 1.5, "tok_max": 2.0}
    assert content_slice([5.0], True) == [5.0]                          # only EOS: keep it
    samples = [{"token_logprob": [-1.0, -3.0, -0.1], "ends_with_eos": True},
               {"token_logprob": [-2.0], "ends_with_eos": False}]
    assert sequence_nll(samples) == pytest.approx((2.0 + 2.0) / 2)


def test_unique_answers_and_clusters():
    reps, idx = unique_answers(["Paris", "paris.", "Lyon", "Unknown", "The Paris"], [False, False, False, True, False])
    assert reps == ["Paris", "Lyon"] and idx == [0, 0, 1, -1, 0]
    same = {("Paris", "Paris City"), ("Paris City", "Paris")}
    answers = ["Paris", "Lyon", "Paris City"]
    ent = lambda i, j: (answers[i], answers[j]) in same
    assert cluster_by_entailment(3, ent) == [0, 1, 0]
    one_way = lambda i, j: (i, j) == (2, 0)                              # not bidirectional
    assert cluster_by_entailment(3, one_way) == [0, 1, 2]


def test_semantic_entropy():
    h, k, per = semantic_entropy([0] * 10, [0])
    assert h == 0.0 and k == 1 and not math.copysign(1, h) < 0          # no negative zero
    h, k, per = semantic_entropy([0] * 5 + [1] * 5, [0, 1])
    assert h == pytest.approx(math.log(2)) and k == 2
    h, k, per = semantic_entropy([0, 0, -1, -1], [0])                   # abstainers = own cluster
    assert k == 2 and per == [0, 0, 1, 1] and h == pytest.approx(math.log(2))
    h, k, _ = semantic_entropy([-1, -1], [])                            # all abstain
    assert h == 0.0 and k == 1


def test_lens_and_attention_features():
    H = np.arange(37, dtype=float)
    top1 = np.zeros(37, bool)
    top1[[20, 30, 36]] = True
    f = lens_features(H, top1, [18, 36])
    assert f == {"lens_H_18": 18.0, "lens_H_36": 36.0, "lens_lstar": 20.0}
    assert lens_features(H, np.zeros(37, bool), [18])["lens_lstar"] == 37.0
    heads = np.zeros((36, 4))
    heads[18] = [0.1, 0.5, 0.2, 0.2]
    a = attention_features(heads, [18])
    assert a == {"attn_mean_18": pytest.approx(0.25), "attn_max_18": 0.5}
    assert math.isnan(attention_features(np.full((36, 4), np.nan), [18])["attn_mean_18"])


# ----------------------------------------------------------------------------- stage

class FakeNLI:
    """Entails iff the normalized answers (after the question prefix) are equal or one contains the other."""

    def __init__(self, *a, **k):
        self.device, self.id2label, self.n_pairs = "cpu", {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}, 0

    def entails(self, pairs):
        self.n_pairs += len(pairs)
        return [h.split("? ")[-1].lower() in p.split("? ")[-1].lower()
                or p.split("? ")[-1].lower() in h.split("? ")[-1].lower() for p, h in pairs]

    def sanity(self):
        return {"same_entails": True, "different_not": True, "paraphrase_entails": True}


def _fixture(tmp_path, n=12, with_folds=True):
    run = tmp_path / "out" / "r1"
    (run / "generation").mkdir(parents=True)
    (run / "correctness").mkdir(parents=True)
    gens, labels = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        correct = i % 3 == 0
        texts = ["Paris"] * 10 if correct else ["Lyon", "Nice", "Lille", "Paris", "Unknown", "Nice", "Metz", "Brest", "Lyon", "Pau"]
        gens.append({"qid": f"q{i}", "question": f"Where is X{i}?",
                     "greedy": {"text": texts[0], "token_entropy": [0.2 if correct else 2.0, 0.1], "token_logprob": [-0.1, -0.01],
                                "ends_with_eos": True},
                     "samples": [{"text": t, "token_logprob": [-0.1 if correct else -2.0, -0.01], "ends_with_eos": True} for t in texts]})
        lab = {"qid": f"q{i}", "label": "correct" if correct else "wrong", "relation": "capital" if i < 6 else "director",
               "relation_group": "closed", "exclude_from_training": False, "dataset": "popqa",
               "sample_labels": ["abstain" if t == "Unknown" else "wrong" for t in texts],
               "abstain_frac": texts.count("Unknown") / 10}
        if with_folds:
            lab["fold"] = i % 5
        labels.append(lab)
    (run / "generation" / "generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens))
    (run / "correctness" / "labels.jsonl").write_text("\n".join(json.dumps(l) for l in labels))
    lens_H = rng.random((n, 37)).astype(np.float32)
    lens_H[:, -1] = [g["greedy"]["token_entropy"][0] for g in gens]
    top1 = np.zeros((n, 37), bool)
    top1[:, 30:] = True
    heads = rng.random((n, 36, 16)).astype(np.float32)
    heads[1] = np.nan                                                    # subject not found
    np.savez(run / "generation" / "features.npz", lens_entropy_tbg=lens_H, lens_target_top1_tbg=top1,
             attn_subj_heads_tbg=heads)
    return {"run_name": "r1", "output_root": str(tmp_path / "out"), "seed": 0, "milestone": "M2",
            "model": {"id": "Qwen/Qwen2.5-3B-Instruct"}, "data": {"dataset": "popqa"},
            "entropy": {"subset": None, "min_per_class": 2, "checkpoint_every": 5}}


def test_stage_end_to_end(tmp_path, monkeypatch):
    from hallu_hierarchy.entropy import nli, stage
    from hallu_hierarchy.reporting.report import Report

    monkeypatch.setattr(nli, "NLI", FakeNLI)
    cfg = _fixture(tmp_path)
    rep = Report(cfg, ["entropy"])
    part = rep.stage("entropy")
    stage.run(cfg, part)
    rep.finish_stage(part)
    rep.write()

    out = tmp_path / "out" / "r1" / "entropy"
    feats = [json.loads(l) for l in (out / "features.jsonl").read_text().splitlines()]
    assert len(feats) == 12 and (out / "features.csv").exists() and (out / "clusters.jsonl").exists()
    f0, f2 = feats[0], feats[2]                     # correct (all "Paris") vs wrong (varied answers)
    assert f0["sem_entropy"] == 0.0 and f0["sem_n_clusters"] == 1
    assert f2["sem_entropy"] > 1.0 and f2["abstain_frac"] == 0.1
    assert f0["tok_mean"] == pytest.approx(0.2) and f0["fold"] == 0
    assert f0["lens_lstar"] == 30.0 and math.isnan(feats[1]["attn_mean_18"])
    s = json.loads((out / "summary.json").read_text())
    assert s["features"]["sem_entropy"]["auroc_wrong"] == 1.0
    assert s["features"]["sem_entropy"]["n_relations_within"] == 2
    text = (tmp_path / "out" / "REPORT.md").read_text()
    assert "✅ logit lens at the last hidden-state index reproduces" in text
    assert "✅ labels with folds produced in this run" in text
    assert "✅ no NaN/inf (attention NaN only where the subject was not found: 1)" in text
    assert "entropy levels: Spearman correlation | VIF" in text
    assert "✅ NLI sanity: same answer entails, different city does not" in text

    # resume: rerun uses the NLI checkpoints (no new pairs)
    monkeypatch.setattr(nli, "NLI", lambda *a, **k: (_ for _ in ()).throw(AssertionError("NLI reloaded")))
    stage.run(cfg, None)


def test_stage_refuses_labels_without_folds(tmp_path, monkeypatch):
    from hallu_hierarchy.entropy import nli, stage

    monkeypatch.setattr(nli, "NLI", FakeNLI)
    cfg = _fixture(tmp_path, with_folds=False)
    with pytest.raises(RuntimeError, match="pre-M1.6 labels"):
        stage.run(cfg, None)
