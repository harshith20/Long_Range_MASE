"""Tests for the M7 synthetic data generator (E20)."""
import copy
import hashlib
import re

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("tokenizers")

from hallu_hierarchy.m7_synthetic.data.generate import (ATTRS, build_people, generate, load_config,
                                                        load_templates)


def small_cfg():
    cfg = load_config("hallu_hierarchy/m7_synthetic/data/config_v0.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["factors"].update({"exp_birth": [1, 10], "ratio": [0, 3], "work_cap": 20,
                           "familiarity": {"low": 2, "high": 5}, "people_per_cell_split": 3})
    cfg["lexicon"].update({"n_cities": 30, "n_universities": 10, "n_employers": 12})
    cfg["tokenizer"]["vocab_size"] = 600
    cfg["output"]["shuffle_buckets"] = 4
    return cfg


def test_templates_one_name_one_fact():
    tpl = load_templates()
    for a in ATTRS:
        for split in ("train", "heldout"):
            for t in tpl["statements"][a][split]:
                assert t.count("{name}") == 1, t
        assert 30 <= len(tpl["statements"][a]["train"]) <= 50
        assert len(tpl["statements"][a]["heldout"]) == 5
        assert not set(tpl["statements"][a]["train"]) & set(tpl["statements"][a]["heldout"])
        assert not set(tpl["questions"][a]["train"]) & set(tpl["questions"][a]["heldout"])


def test_people_constraints():
    people, meta = build_people(small_cfg())
    assert len(people) == 2 * 2 * 2 * 2 * 3
    assert len({p["name"] for p in people}) == len(people)
    for p in people:
        assert len({p["birth_city"], p["work_city"], p["university_city"], p["employer_city"]}) == 4
        assert p["n_work_city"] == min(p["exp_birth"] * p["ratio"], 20)
    assert any(p["capped"] for p in people)  # 10 x 3 = 30 > cap 20


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("m7")
    cfg = small_cfg()
    budget = generate(cfg, load_templates(), out, fraction=1.0, quiet=True)
    return out, budget, cfg


def test_counts_and_qa(run):
    out, budget, cfg = run
    P = pd.read_parquet(out / "people.parquet")
    L = pd.read_parquet(out / "exposure_log.parquet")
    E = L[L.is_exposure]
    got = E.groupby(["person_id", "attribute"]).size()
    for _, r in P.iterrows():
        for a in ATTRS:
            assert got.get((r.person_id, a), 0) == r[f"n_{a}"]
    Q = L[~L.is_exposure].merge(P[["person_id", "split", "ratio"]], on="person_id")
    assert set(Q.split) == {"qa_train"}
    assert not ((Q.ratio == 0) & (Q.attribute == "work_city")).any()
    # name balancing: equal name mentions across ratios within (exp_birth, familiarity), up to anchor mentions
    spread = P.groupby(["exp_birth", "familiarity"]).name_mentions.agg(lambda s: s.max() - s.min())
    assert (spread <= cfg["cooccurrence"]["max_partners"] * cfg["cooccurrence"]["cross_max_per_person"] + 2).all()


def test_heldout_templates_never_in_corpus(run):
    out, _, _ = run
    tpl = load_templates()
    text = (out / "corpus.jsonl").read_text(encoding="utf-8")
    for a in ATTRS:
        for t in tpl["statements"][a]["heldout"]:
            # the fixed words of the template, with placeholders as wildcards
            pat = re.escape(t)
            pat = re.sub(r"\\\{\w+\\\}", r".+?", pat)
            assert not re.search(pat, text), t


def test_tokens_and_offsets(run):
    import numpy as np
    out, budget, _ = run
    D = pd.read_parquet(out / "documents.parquet")
    tok = np.fromfile(out / "tokens_000.bin", dtype=np.uint16)
    assert len(tok) == D.n_tokens.sum() == budget["measured_this_run"]["tokens_total"]
    assert (tok[D.token_offset + D.n_tokens - 1] == budget["eos_id"]).all()


def test_deterministic(tmp_path):
    cfg = small_cfg()
    h = []
    for i in range(2):
        out = tmp_path / f"r{i}"
        generate(cfg, load_templates(), out, fraction=1.0, quiet=True)
        h.append(hashlib.md5((out / "tokens_000.bin").read_bytes()).hexdigest())
    assert h[0] == h[1]


def test_refuses_overwrite(run):
    out, _, cfg = run
    with pytest.raises(FileExistsError):
        generate(cfg, load_templates(), out, fraction=1.0, quiet=True)
