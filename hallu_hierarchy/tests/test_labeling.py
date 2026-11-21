"""CPU tests for M3 labeling: co-occurrence logic, the cached Wikipedia client and the stage
(fake API, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hallu_hierarchy.labeling.cooc import answer_candidates, answer_status, final_label, is_ah, rule_names
from hallu_hierarchy.labeling.wiki import WikiCache, WikiClient, base_title, clean_wikitext

HW = ("tv series", "film", "series")

PAGES = {
    "Carrie (1976 film)": "'''Carrie''' is a 1976 film directed by [[Brian De Palma]].<ref>Alex Kurtzman interview</ref> "
                          "Music by [[Pino Donaggio]]. Genre: [[horror film|horror]].",
    "Carrie": "'''Carrie''' may refer to: Carrie (2013 film), directed by Kimberly Peirce; produced by Kevin Misher",
    "Remco Campert": "Remco Campert is a Dutch writer, son of [[Jan Campert]].",
}
REDIRECTS = {"Carrie (disambiguation)": "Carrie"}
DISAMBIG = {"Carrie"}
HITS = {('"brian de palma" linksto:"Carrie (1976 film)"'): 88, ('"kevin misher" linksto:"Carrie (1976 film)"'): 3}


def fake_get(params: dict) -> dict:
    if params.get("list") == "search":
        return {"query": {"searchinfo": {"totalhits": HITS.get(params["srsearch"], 0)}}}
    titles = params["titles"].split("|")
    q: dict = {"redirects": [{"from": t, "to": REDIRECTS[t]} for t in titles if t in REDIRECTS], "pages": []}
    for t in {REDIRECTS.get(t, t) for t in titles}:
        if t in PAGES:
            p = {"title": t, "revisions": [{"slots": {"main": {"content": PAGES[t]}}}]}
            if t in DISAMBIG:
                p["pageprops"] = {"disambiguation": ""}
            q["pages"].append(p)
        else:
            q["pages"].append({"title": t, "missing": True})
    return {"query": q}


# ----------------------------------------------------------------------------- pure logic

def test_clean_wikitext_drops_refs_keeps_links():
    t = clean_wikitext(PAGES["Carrie (1976 film)"])
    assert "brian de palma" in t and "horror film horror" in t
    assert "kurtzman" not in t                       # reference text is dropped


def test_base_title():
    assert base_title("Carrie (1976 film)") == "Carrie"
    assert base_title("Hun, Libya") == "Hun, Libya"


def test_answer_candidates_and_status():
    assert answer_candidates("action film", HW) == ["action film", "action"]
    assert answer_candidates("x", HW) == []
    assert answer_status("brian de palma", "Carrie", "Carrie (1976 film)") == "ok"
    assert answer_status("remco campert has no known father", "Remco Campert", "Remco Campert") == "sentence"
    assert answer_status("hun remains unspecified", "Hun", "Hun, Libya") == "sentence"
    assert answer_status("a b c d e f", None, None) == "sentence"
    assert answer_status("", None, None) == "empty"
    assert answer_status("carrie white", "Carrie", "Carrie (1976 film)") == "ok"   # 2 words: a name, not a sentence


def test_rules_and_final_label():
    assert rule_names([1, 5]) == ["page", "count>=1", "count>=5"]
    assert is_ah("page", True, None) is True
    assert is_ah("count>=5", False, None) is None
    assert is_ah("count>=5", False, 7) is True
    base = {"label": "wrong", "exclude_from_training": False, "m3_status": "ok"}
    assert final_label({**base, "ah": {"page": True}, "disambig_exists": False}, "page", "disambig_exists") == "AH"
    assert final_label({**base, "ah": {"page": True}, "disambig_exists": True}, "page", "disambig_exists") == "ambiguous_confirmed"
    assert final_label({**base, "ah": {"page": False}, "disambig_exists": True}, "page", "disambig_exists") == "UH"
    assert final_label({**base, "m3_status": "sentence", "ah": {"page": None}}, "page", "disambig_exists") == "unscored"
    assert final_label({**base, "label": "correct", "ah": {}}, "page", "disambig_exists") == "FA"
    assert final_label({**base, "label": "abstain", "exclude_from_training": True, "ah": {}}, "page", "x") == "abstain"
    assert final_label({**base, "label": "partial", "exclude_from_training": True, "ah": {}}, "page", "x") == "excluded"


# ----------------------------------------------------------------------------- client + cache

def test_client_redirects_disambig_and_cache(tmp_path: Path):
    calls = []

    def get(p):
        calls.append(p)
        return fake_get(p)

    cache = WikiCache(tmp_path / "w.sqlite")
    c = WikiClient(cache, interval_s=0, get=get)
    pg = c.pages(["Carrie (1976 film)", "Carrie (disambiguation)", "Nope"])
    assert pg["Carrie (disambiguation)"]["canonical"] == "Carrie" and pg["Carrie (disambiguation)"]["disambig"]
    assert pg["Nope"]["missing"] and not pg["Carrie (1976 film)"]["missing"]
    assert c.linked_hits("brian de palma", "Carrie (1976 film)") == 88
    n = len(calls)
    c.pages(["Carrie (1976 film)", "Nope"])
    c.linked_hits("brian de palma", "Carrie (1976 film)")
    assert len(calls) == n                           # second time: all from cache
    cache.close()


# ----------------------------------------------------------------------------- stage

def _write_inputs(root: Path) -> None:
    rd = root / "t_run"
    (rd / "correctness").mkdir(parents=True)
    (rd / "data").mkdir(parents=True)

    def lab(qid, label, pred, aliases, subject, relation="director", excl=False):
        return {"qid": qid, "dataset": "popqa", "fold": 0, "relation": relation, "relation_group": "open",
                "s_pop": 1, "subject": subject, "subject_id": subject, "question": "q?", "pred": pred,
                "pred_norm": None, "aliases": aliases, "label": label, "exclude_from_training": excl,
                "flags": {"ambiguous_subject": False, "gold_noise": False}}
    rows = [
        lab("q1", "wrong", "Kevin Misher", ["Brian De Palma"], "Carrie"),        # on disambig page, count 3
        lab("q2", "wrong", "Pino Donaggio", ["Brian De Palma"], "Carrie"),       # on subject page -> AH (ambiguous)
        lab("q3", "wrong", "John Smith", ["Brian De Palma"], "Carrie"),          # nowhere -> UH
        lab("q4", "correct", "Brian De Palma", ["Brian De Palma"], "Carrie"),    # FA
        lab("q5", "wrong", "Remco Campert has no known father.", ["Jan Campert"], "Remco Campert", "father"),
        lab("q6", "abstain", "unknown", ["x"], "Remco Campert", "father", excl=True),
    ]
    titles = {"q1": "Carrie (1976 film)", "q2": "Carrie (1976 film)", "q3": "Carrie (1976 film)",
              "q4": "Carrie (1976 film)", "q5": "Remco Campert", "q6": "Remco Campert"}
    with open(rd / "correctness" / "labels.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(rd / "data" / "questions.jsonl", "w") as f:
        f.writelines(json.dumps({"qid": q, "meta": {"s_wiki_title": t}}) + "\n" for q, t in titles.items())


def test_stage_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hallu_hierarchy.labeling import stage, wiki
    from hallu_hierarchy.reporting.report import Report

    monkeypatch.setattr(wiki, "http_get_json", fake_get)
    monkeypatch.setattr(stage.WikiClient.__init__, "__defaults__", (0.1, fake_get))
    _write_inputs(tmp_path)
    cfg = {"run_name": "t_run", "seed": 0, "output_root": str(tmp_path), "data": {"dataset": "popqa"},
           "model": {"id": "Qwen/Qwen2.5-3B-Instruct"},
           "labeling": {"request_interval_s": 0, "count_cutoffs": [1, 5], "manual_check_n": 4}}
    rep = Report(cfg, ["labeling"])
    part = rep.stage("labeling")
    stage.run(cfg, part)
    rows = {json.loads(l)["qid"]: json.loads(l) for l in open(tmp_path / "t_run" / "labeling" / "labels_m3.jsonl")}

    assert rows["q2"]["cooc_page"] and rows["q2"]["disambig_exists"] and not rows["q2"]["answer_on_disambig"]
    assert rows["q2"]["label_m3"] == "ambiguous_confirmed"           # default rule: disambig_exists
    assert rows["q1"]["cooc_page"] is False and rows["q1"]["cooc_count"] == 3 and rows["q1"]["answer_on_disambig"]
    assert rows["q1"]["label_m3_by_rule"] == {"page": "UH", "count>=1": "ambiguous_confirmed", "count>=5": "UH"}
    assert rows["q3"]["label_m3"] == "UH" and rows["q3"]["cooc_count"] == 0
    assert rows["q3"]["gold_on_page"] and rows["q3"]["gold_count"] == 89   # 88 linked + subject page
    assert rows["q4"]["label_m3"] == "FA"
    assert rows["q5"]["m3_status"] == "sentence" and rows["q5"]["label_m3"] == "unscored"
    assert rows["q6"]["label_m3"] == "abstain"
    s = part.summary
    assert s["labels_have_folds"] and s["ah_before_after"]["page"] == {
        "before": 1, "after_disambig_exists": 0, "after_answer_on_disambig": 1}
    text = rep.render()
    assert "labels per rule" in text and "sentences" in text           # action list for the sentence answer
    assert (tmp_path / "t_run" / "labeling" / "manual_check.csv").exists()

    # rerun: everything from the cache, no API calls
    monkeypatch.setattr(stage.WikiClient.__init__, "__defaults__", (0.1, lambda p: pytest.fail("network used")))
    stage.run(cfg, None)
