"""Stage `labeling` (M3): AH / UH labels for wrong answers via the Wikipedia proxy (CLAUDE.md §3).

CPU only, needs internet the first time (everything is cached in SQLite; reruns are offline).

For each training-eligible wrong answer (and each eligible correct answer, as the FA reference):
  1. fetch the subject's Wikipedia page (PopQA `s_wiki_title`, redirects followed)
  2. cooc_page:   the model's answer appears on that page (whole-word, normalized, head-word-stripped copy too)
     gold_on_page: the same test with the gold aliases — validates the proxy (if the gold answer is
                   rarely on the page, a missing wrong answer says little)
  3. cooc_count = pages that contain the answer and link to the subject page (+1 if on the page);
     gold_count on the same rows for calibration
  4. AH / UH per rule: "page" and "count>=k" for each cutoff (CLAUDE.md §9: report 2–3 cutoffs)
  5. ambiguity check (all scored wrong rows, applied to AH rows only):
       disambig_exists     a disambiguation page exists for the subject's base name
       answer_on_disambig  the model's answer appears on that disambiguation page
                           (= the answer is plausibly true for another entity with the same name)
     AH rows flagged by `ambiguity_rule` become `ambiguous_confirmed` and leave AH.
Wrong answers that restate the subject or are long sentences (mostly missed refusals) are not
scored ('unscored') and listed for the refusal-pattern list.

Outputs (outputs/<run>/labeling/): labels_m3.jsonl, labels_m3.csv, manual_check.csv,
sentence_answers.csv, summary.json, meta.json. Cache: <output_root>/wiki_cache/wiki.sqlite.
"""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..correctness.matching import load_head_words, strip_head_words
from ..correctness.review import load_relation_groups, write_csv
from ..data.normalize import normalize_answer
from ..reporting.console import model_short, pct, table
from ..utils import (
    PKG_DIR, default_output_root, find_upstream, read_jsonl, resolve_pkg_path, run_dir, write_json,
    write_jsonl, write_meta,
)
from .cooc import answer_candidates, answer_status, any_on, final_label, is_ah, rule_names
from .wiki import WikiCache, WikiClient, base_title

DEFAULTS: dict[str, Any] = {
    "subset": None,               # {wrong: n, correct: n} seeded sample for a dev run; null = all
    "linked_counts": True,        # cooc_count via `linksto:` search (one request per wrong answer)
    "count_cutoffs": [1, 3, 10],  # AH if cooc_count >= k
    "primary_rule": "page",       # rule for `label_m3` until the manual check decides
    "ambiguity_rule": "disambig_exists",   # or answer_on_disambig (CLAUDE.md §9 open decision)
    "gold_count_sample": 1000,    # wrong rows that also get gold_count (all rows in a subset run)
    "count_correct": True,        # cooc_count for correct answers too (FA reference)
    "request_interval_s": 0.1,
    "cache_dir": None,            # default <output_root>/wiki_cache
    "head_words": "configs/head_words.yaml",
    "relation_groups": "configs/relation_groups.yaml",
    "manual_check_n": 100,
    "min_page_found": 0.95,
    "min_gold_on_page": 0.5,
    "max_unscored": 0.05,
    "min_per_class": 30,
}
WIKI_URL = "https://en.wikipedia.org/wiki/"


def _cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULTS, **(cfg.get("labeling") or {})}


def manual_path(cfg: dict[str, Any]) -> Path:
    return PKG_DIR / "overrides" / f"m3_manual_{cfg['data']['dataset']}_{model_short(cfg)}.csv"


# ----------------------------------------------------------------------------- selection

def select_rows(labels: list[dict[str, Any]], subset: dict[str, int] | None, seed: int) -> list[dict[str, Any]]:
    """All rows, or a seeded sample of eligible wrong + eligible correct rows (dev run)."""
    if not subset:
        return labels
    rng = random.Random(seed)
    out = []
    for lab in ("wrong", "correct"):
        pool = [r for r in labels if r["label"] == lab and not r["exclude_from_training"]]
        out += rng.sample(pool, min(int(subset.get(lab, 0)), len(pool)))
    return sorted(out, key=lambda r: r["qid"])


def stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """~n scored wrong rows, balanced over (AH/UH under the primary rule) x (open/closed)."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["label_m3"] in ("AH", "UH", "ambiguous_confirmed"):
            cells[(r["label_m3"], r["relation_group"] or "-")].append(r)
    rng = random.Random(seed)
    keys = sorted(cells)
    out: list[dict[str, Any]] = []
    per = max(1, n // max(1, len(keys)))
    for k in keys:
        out += rng.sample(cells[k], min(per, len(cells[k])))
    taken = {r["qid"] for r in out}
    rest = [r for k in keys for r in cells[k] if r["qid"] not in taken]
    out += rng.sample(rest, min(max(0, n - len(out)), len(rest)))
    return sorted(out, key=lambda r: r["qid"])


# ----------------------------------------------------------------------------- scoring

def score_rows(rows: list[dict[str, Any]], titles: dict[str, str], client: WikiClient,
               lc: dict[str, Any], head_words: tuple[str, ...], seed: int) -> dict[str, Any]:
    """Adds the M3 columns to `rows` in place; returns timing / request stats."""
    t0 = time.perf_counter()
    scored = [r for r in rows if r["label"] in ("wrong", "correct") and not r["exclude_from_training"]]
    log = lambda msg: print(f"[labeling] {msg}", flush=True)  # noqa: E731

    log(f"subject pages for {len(scored)} rows ({len({titles[r['qid']] for r in scored})} titles)")
    pages = client.pages([titles[r["qid"]] for r in scored],
                         progress=lambda i, n: log(f"  pages {i}/{n}") if i % 1000 < 50 or i == n else None)
    t_pages = time.perf_counter() - t0

    for r in scored:
        title = titles[r["qid"]]
        pg = pages.get(title) or {"missing": True, "canonical": title, "text": "", "disambig": False}
        r["s_wiki_title"], r["wiki_canonical"] = title, pg["canonical"]
        r["page_found"] = not pg["missing"]
        r["page_words"] = len(pg["text"].split())
        r["page_is_disambig"] = bool(pg.get("disambig"))
        pred_norm = r.get("pred_norm") or normalize_answer(r["pred"])
        r["answer_norm"] = pred_norm
        status = answer_status(pred_norm, r.get("subject"), title)
        r["m3_status"] = status if r["page_found"] else "no_page"
        cands = answer_candidates(pred_norm, head_words)
        gold = [normalize_answer(a) for a in r["aliases"]]
        gold_c = list(dict.fromkeys(g for a in gold for g in (a, strip_head_words(a, head_words)) if g and len(g) >= 2))
        r["cooc_page"] = any_on(pg["text"], cands) if r["m3_status"] == "ok" else None
        r["gold_on_page"] = any_on(pg["text"], gold_c) if r["page_found"] else None
        r["gold_on_page_alias"] = next((g for g in gold_c if any_on(pg["text"], [g])), None)
        r["linked_hits"] = r["cooc_count"] = r["gold_count"] = None
        r["disambig_exists"] = r["answer_on_disambig"] = None

    wrong_ok = [r for r in scored if r["label"] == "wrong" and r["m3_status"] == "ok"]
    corr_ok = [r for r in scored if r["label"] == "correct" and r["m3_status"] == "ok"]
    t1 = time.perf_counter()
    if lc["linked_counts"]:
        todo = wrong_ok + (corr_ok if lc["count_correct"] else [])
        log(f"linked counts for {len(todo)} answers (one search each, cached)")
        for i, r in enumerate(todo, 1):
            r["linked_hits"] = client.linked_hits(r["answer_norm"], r["wiki_canonical"])
            r["cooc_count"] = r["linked_hits"] + int(bool(r["cooc_page"]))
            if i % 500 == 0 or i == len(todo):
                log(f"  counts {i}/{len(todo)} ({client.n_requests} requests so far)")
        n_gold = len(wrong_ok) if lc["subset"] else min(int(lc["gold_count_sample"]), len(wrong_ok))
        gold_rows = random.Random(seed).sample(wrong_ok, n_gold)
        log(f"gold counts for {len(gold_rows)} wrong rows (calibration)")
        for i, r in enumerate(gold_rows, 1):
            alias = r["gold_on_page_alias"] or normalize_answer(r["aliases"][0])
            r["gold_count"] = client.linked_hits(alias, r["wiki_canonical"]) + int(bool(r["gold_on_page"]))
            if i % 500 == 0 or i == len(gold_rows):
                log(f"  gold counts {i}/{len(gold_rows)}")
    t_counts = time.perf_counter() - t1

    # Ambiguity: disambiguation page for the subject's base name ("Carrie (1976 film)" -> "Carrie").
    t2 = time.perf_counter()
    def dis_titles(r: dict[str, Any]) -> list[str]:
        b = base_title(r["s_wiki_title"])
        return [f"{b} (disambiguation)"] + ([b] if b != r["s_wiki_title"] else [])
    dpages = client.pages([t for r in wrong_ok for t in dis_titles(r)],
                          progress=lambda i, n: log(f"  disambiguation pages {i}/{n}") if i % 1000 < 50 or i == n else None)
    for r in wrong_ok:
        texts = [dpages[t]["text"] for t in dis_titles(r) if t in dpages and dpages[t]["disambig"]]
        r["disambig_exists"] = bool(texts)
        r["answer_on_disambig"] = any(any_on(t, answer_candidates(r["answer_norm"], head_words)) for t in texts)
    t_dis = time.perf_counter() - t2

    rules = rule_names(lc["count_cutoffs"])
    for r in rows:
        r.setdefault("m3_status", "not_scored")
        r["ah"] = {rule: (is_ah(rule, bool(r.get("cooc_page")), r.get("cooc_count"))
                          if r["label"] == "wrong" and r["m3_status"] == "ok" else None) for rule in rules}
        r["label_m3_by_rule"] = {rule: final_label(r, rule, lc["ambiguity_rule"]) for rule in rules}
        r["label_m3"] = r["label_m3_by_rule"][lc["primary_rule"]]
    return {"seconds": {"pages": t_pages, "counts": t_counts, "disambig": t_dis, "total": time.perf_counter() - t0},
            "requests": client.n_requests, "request_seconds": client.seconds}


# ----------------------------------------------------------------------------- manual check

def load_manual(path: Path) -> dict[str, str]:
    """qid -> manual label (AH / UH / not_wrong / unclear) from the committed manual-check file."""
    import csv
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r["manual_label"].strip() for r in csv.DictReader(f) if (r.get("manual_label") or "").strip()}


def manual_agreement(rows: list[dict[str, Any]], manual: dict[str, str], rules: list[str]) -> dict[str, Any]:
    by_qid = {r["qid"]: r for r in rows}
    usable = {q: m for q, m in manual.items() if m in ("AH", "UH") and q in by_qid}
    out: dict[str, Any] = {"n_manual": len(manual), "n_ah_uh": len(usable),
                           "manual_counts": dict(Counter(manual.values()))}
    for rule in rules:
        pairs = [(by_qid[q]["ah"].get(rule), m == "AH") for q, m in usable.items() if by_qid[q]["ah"].get(rule) is not None]
        if pairs:
            out[rule] = {"n": len(pairs), "agree": sum(a == b for a, b in pairs) / len(pairs),
                         "ah_precision": (sum(a and b for a, b in pairs) / max(1, sum(a for a, _ in pairs))),
                         "ah_recall": (sum(a and b for a, b in pairs) / max(1, sum(b for _, b in pairs)))}
    return out


# ----------------------------------------------------------------------------- summary

def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    v = [r[key] for r in rows if r.get(key) is not None]
    return sum(map(bool, v)) / len(v) if v else None


def _quantiles(v: list[int]) -> list[int] | None:
    if not v:
        return None
    s = sorted(v)
    return [s[int(q * (len(s) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)]


def summarize(rows: list[dict[str, Any]], lc: dict[str, Any], groups: dict[str, str],
              borderline: set[str], stats: dict[str, Any], manual: dict[str, str]) -> dict[str, Any]:
    rules = rule_names(lc["count_cutoffs"])
    wrong = [r for r in rows if r["label"] == "wrong" and not r["exclude_from_training"]]
    wok = [r for r in wrong if r["m3_status"] == "ok"]
    corr = [r for r in rows if r["label"] == "correct" and not r["exclude_from_training"]]
    wrong_ids = {r["qid"] for r in wrong}
    s: dict[str, Any] = {
        "n": len(rows), "n_wrong_eligible": len(wrong), "n_wrong_scored": len(wok), "n_correct_eligible": len(corr),
        "m3_status": dict(Counter(r["m3_status"] for r in wrong)),
        "page_found": _rate(wrong + corr, "page_found"),
        "gold_on_page_wrong": _rate(wrong, "gold_on_page"), "gold_on_page_correct": _rate(corr, "gold_on_page"),
        "cooc_page_wrong": _rate(wok, "cooc_page"),
        "cooc_page_correct": _rate([r for r in corr if r["m3_status"] == "ok"], "cooc_page"),
        "page_words_median": _quantiles([r["page_words"] for r in wrong + corr if r.get("page_found")]),
        "count_quantiles": {
            "wrong_answer": _quantiles([r["cooc_count"] for r in wok if r["cooc_count"] is not None]),
            "gold_same_rows": _quantiles([r["gold_count"] for r in wok if r["gold_count"] is not None]),
            "correct_answer": _quantiles([r["cooc_count"] for r in corr if r.get("cooc_count") is not None]),
        },
        "n_gold_count": sum(r["gold_count"] is not None for r in wok),
        "disambig_exists": _rate(wok, "disambig_exists"), "answer_on_disambig": _rate(wok, "answer_on_disambig"),
        "rules": rules, "primary_rule": lc["primary_rule"], "ambiguity_rule": lc["ambiguity_rule"],
        "count_cutoffs": lc["count_cutoffs"], "subset": lc["subset"], **stats,
    }
    s["labels_by_rule"] = {rule: dict(Counter(r["label_m3_by_rule"][rule] for r in rows)) for rule in rules}
    s["ah_before_after"] = {}
    for rule in rules:
        before = sum(1 for r in wok if r["ah"].get(rule))
        s["ah_before_after"][rule] = {
            "before": before,
            "after_disambig_exists": sum(1 for r in wok if r["ah"].get(rule) and not r["disambig_exists"]),
            "after_answer_on_disambig": sum(1 for r in wok if r["ah"].get(rule) and not r["answer_on_disambig"]),
        }
    by_group: dict[str, Any] = {}
    for g in ("closed", "open"):
        gr = [r for r in rows if r.get("relation_group") == g]
        by_group[g] = {rule: dict(Counter(r["label_m3_by_rule"][rule] for r in gr)) for rule in rules}
        by_group[g]["gold_on_page_wrong"] = _rate([r for r in gr if r["qid"] in wrong_ids], "gold_on_page")
    s["by_group"] = by_group
    rel: dict[str, Any] = {}
    for name in sorted({r["relation"] for r in wrong}):
        rw = [r for r in wok if r["relation"] == name]
        rel[name] = {"group": groups.get(name, "-"), "borderline": name in borderline,
                     "n_wrong": sum(r["relation"] == name for r in wrong), "n_scored": len(rw),
                     "gold_on_page": _rate([r for r in wrong if r["relation"] == name], "gold_on_page"),
                     **{f"ah_{rule}": (sum(1 for r in rw if r["ah"].get(rule)) / len(rw) if rw else None) for rule in rules},
                     "disambig": _rate(rw, "disambig_exists"),
                     "n_fa": sum(r["relation"] == name for r in corr)}
    s["by_relation"] = rel
    s["manual"] = manual_agreement(rows, manual, rules) if manual else None
    return s


# ----------------------------------------------------------------------------- report

def _p(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def fill_report(part: Any, s: dict[str, Any], lc: dict[str, Any], rows: list[dict[str, Any]], mpath: Path) -> None:
    part.n, part.summary = s["n"], s
    rules = s["rules"]
    nw, nok = s["n_wrong_eligible"], s["n_wrong_scored"]
    part.hard(s["labels_have_folds"], "labels are the M1.6 labels with folds" if s["labels_have_folds"]
              else "labels have no `fold` column (pre-M1.6): run `correctness` first")
    pf = s["page_found"] or 0.0
    part.soft(pf >= lc["min_page_found"], f"subject Wikipedia page found for {_p(pf)} of scored rows"
              + ("" if pf >= lc["min_page_found"] else " — titles missing on Wikipedia are left unscored"))
    g = s["gold_on_page_wrong"] or 0.0
    part.soft(g >= lc["min_gold_on_page"],
              f"proxy check: the GOLD answer is on the subject page for {_p(g)} of wrong rows"
              + ("" if g >= lc["min_gold_on_page"] else
                 " — below 50%: pages often miss even the true fact, so 'answer not on page' (UH) is weak "
                 "evidence; prefer a count rule"))
    uns = 1 - nok / nw if nw else 0.0
    part.soft(uns <= lc["max_unscored"], f"unscored wrong answers {pct(nw - nok, nw)} ({s['m3_status']})"
              + ("" if uns <= lc["max_unscored"] else " — mostly sentence answers (missed refusals); see list"))
    lab = s["labels_by_rule"][s["primary_rule"]]
    for c in ("FA", "AH", "UH"):
        part.soft(lab.get(c, 0) >= lc["min_per_class"], f"{c} has {lab.get(c, 0)} rows under rule '{s['primary_rule']}'"
                  + ("" if lab.get(c, 0) >= lc["min_per_class"] else f" (< {lc['min_per_class']}: too few to evaluate)"))

    part.key.append(f"rows {s['n']} | eligible wrong {nw} (scored {nok}) | eligible correct (FA) {s['n_correct_eligible']}"
                    + (f" | SUBSET {s['subset']}" if s["subset"] else ""))
    part.key.append(f"answer on subject page: wrong {_p(s['cooc_page_wrong'])} | correct {_p(s['cooc_page_correct'])} || "
                    f"GOLD on page: wrong rows {_p(s['gold_on_page_wrong'])} | correct rows {_p(s['gold_on_page_correct'])}")
    cq = s["count_quantiles"]
    if cq["wrong_answer"]:
        part.key.append(f"cooc_count p10/25/50/75/90: wrong answer {cq['wrong_answer']} | gold, same rows "
                        f"{cq['gold_same_rows']} (n={s['n_gold_count']}) | correct answers {cq['correct_answer']}")
    part.key.append("labels per rule (AH/UH among wrong; ambiguous = AH removed by the ambiguity check)")
    cols = ["FA", "AH", "UH", "ambiguous_confirmed", "unscored", "abstain", "excluded"]
    part.key += table([["rule", *[c.replace("ambiguous_confirmed", "ambig") for c in cols], "AH share"]] + [
        [rule + (" *" if rule == s["primary_rule"] else ""), *[s["labels_by_rule"][rule].get(c, 0) for c in cols],
         _p(s["labels_by_rule"][rule].get("AH", 0) / max(1, s["labels_by_rule"][rule].get("AH", 0) + s["labels_by_rule"][rule].get("UH", 0)))]
        for rule in rules])
    part.key.append(f"ambiguity check on wrong rows: disambiguation page {_p(s['disambig_exists'])} | answer on it "
                    f"{_p(s['answer_on_disambig'])} | rule in use: {s['ambiguity_rule']}")
    part.key += table([["AH before → after", "before", "disambig_exists", "answer_on_disambig"]] + [
        [rule, v["before"], v["after_disambig_exists"], v["after_answer_on_disambig"]] for rule, v in s["ah_before_after"].items()])
    part.key.append(f"by group, rule '{s['primary_rule']}'")
    part.key += table([["group", "FA", "AH", "UH", "ambig", "unscored", "gold on page"]] + [
        [gname, *[gd[s["primary_rule"]].get(c, 0) for c in ("FA", "AH", "UH", "ambiguous_confirmed", "unscored")],
         _p(gd["gold_on_page_wrong"])] for gname, gd in s["by_group"].items()])
    part.key.append("by relation (AH% = share of scored wrong answers labelled AH before the ambiguity check)")
    part.key += table([["relation", "group", "wrong", "gold on page", *[f"AH% {r}" for r in rules], "disambig", "FA"]] + [
        [name, v["group"] + ("*" if v["borderline"] else ""), v["n_wrong"], _p(v["gold_on_page"]),
         *[_p(v[f"ah_{r}"]) for r in rules], _p(v["disambig"]), v["n_fa"]]
        for name, v in sorted(s["by_relation"].items(), key=lambda kv: (kv[1]["group"], -kv[1]["n_wrong"]))])
    if s["manual"]:
        m = s["manual"]
        part.key.append(f"manual check ({m['n_manual']} rows, {m['n_ah_uh']} AH/UH): {m['manual_counts']}")
        part.key += table([["rule", "n", "agree", "AH precision", "AH recall"]] + [
            [r, m[r]["n"], _p(m[r]["agree"]), _p(m[r]["ah_precision"]), _p(m[r]["ah_recall"])] for r in rules if r in m])
    else:
        part.key.append(f"manual check: none yet (fill `manual_label` in manual_check.csv → {mpath.relative_to(PKG_DIR)})")
    sec = s["seconds"]
    part.key.append(f"Wikipedia: {s['requests']} requests this run, cache {s['cache_counts']} | time pages "
                    f"{sec['pages']:.0f}s, counts {sec['counts']:.0f}s, disambiguation {sec['disambig']:.0f}s")

    sent = [r for r in rows if r.get("m3_status") == "sentence"]
    if sent:
        part.action.append(f"{len(sent)} wrong answers are sentences (mostly missed refusals), left unscored; "
                           "candidates for configs/refusal_patterns.yaml (all in labeling/sentence_answers.csv):")
        part.action += [f"  {r['qid']} [{r['relation']}] {r['pred']!r}" for r in sent[:20]]

    part.claude.append(f"n={s['n']} wrong={nw} scored={nok} correct={s['n_correct_eligible']} status={s['m3_status']} subset={s['subset']}")
    part.claude.append(f"page_found={s['page_found']:.4f} gold_on_page wrong={s['gold_on_page_wrong']} correct={s['gold_on_page_correct']} "
                       f"cooc_page wrong={s['cooc_page_wrong']} correct={s['cooc_page_correct']} page_words q={s['page_words_median']}")
    part.claude.append(f"count_quantiles={cq} n_gold_count={s['n_gold_count']}")
    part.claude.append(f"labels_by_rule={s['labels_by_rule']}")
    part.claude.append(f"ah_before_after={s['ah_before_after']} disambig={s['disambig_exists']} answer_on_disambig={s['answer_on_disambig']}")
    part.claude.append(f"by_group={s['by_group']}")
    part.claude.append(f"rules={rules} primary={s['primary_rule']} ambiguity_rule={s['ambiguity_rule']} manual={s['manual']}")
    part.claude.append(f"requests={s['requests']} seconds={ {k: round(v) for k, v in sec.items()} } labels_path={s['labels_path']}")


# ----------------------------------------------------------------------------- run

CSV_COLS = ["qid", "fold", "relation", "relation_group", "subject", "s_wiki_title", "wiki_canonical", "question",
            "pred", "label", "m3_status", "cooc_page", "cooc_count", "gold_on_page", "gold_count",
            "disambig_exists", "answer_on_disambig", "label_m3"]


def run(cfg: dict[str, Any], report: Any = None) -> None:
    lc = _cfg(cfg)
    seed = int(cfg.get("seed", 0))
    out = run_dir(cfg) / "labeling"
    labels_path = find_upstream(cfg, "correctness/labels.jsonl")
    q_path = find_upstream(cfg, "data/questions.jsonl")
    if labels_path is None or q_path is None:
        raise FileNotFoundError("labeling needs correctness/labels.jsonl and data/questions.jsonl "
                                "(run `correctness` first, or attach/point --input-run-dir at them)")
    cfg["input_run_dir_used"] = str(labels_path.parent.parent)
    labels = read_jsonl(labels_path)
    questions = {q["qid"]: q for q in read_jsonl(q_path)}
    titles = {qid: (q.get("meta") or {}).get("s_wiki_title") or q.get("subject") for qid, q in questions.items()}

    rows = select_rows(labels, lc["subset"], seed)
    groups, borderline = load_relation_groups(resolve_pkg_path(lc["relation_groups"]))
    head_words = load_head_words(resolve_pkg_path(lc["head_words"]))
    for r in rows:
        r.setdefault("relation_group", groups.get(str(r["relation"]).lower()))

    cache_path = Path(lc["cache_dir"] or (Path(cfg.get("output_root") or default_output_root()) / "wiki_cache")) / "wiki.sqlite"
    cache = WikiCache(cache_path)
    try:
        client = WikiClient(cache, interval_s=float(lc["request_interval_s"]))
        stats = score_rows(rows, titles, client, lc, head_words, seed)
        stats["cache_counts"] = cache.counts()
    finally:
        cache.close()
    stats["cache_path"] = str(cache_path)

    mpath = manual_path(cfg)
    s = summarize(rows, lc, groups, borderline, stats, load_manual(mpath))
    s["labels_path"] = str(labels_path)
    s["labels_have_folds"] = all(r.get("fold") is not None for r in labels)

    keep = ["qid", "dataset", "fold", "relation", "relation_group", "s_pop", "subject", "subject_id", "question",
            "pred", "aliases", "label", "exclude_from_training", "flags", "s_wiki_title", "wiki_canonical",
            "page_found", "page_words", "page_is_disambig", "answer_norm", "m3_status", "cooc_page",
            "gold_on_page", "gold_on_page_alias", "linked_hits", "cooc_count", "gold_count", "disambig_exists",
            "answer_on_disambig", "ah", "label_m3_by_rule", "label_m3"]
    write_jsonl(out / "labels_m3.jsonl", [{k: r.get(k) for k in keep} for r in rows])
    write_csv(out / "labels_m3.csv", CSV_COLS, [[r.get(k) for k in CSV_COLS] for r in rows])
    sample = stratified_sample(rows, int(lc["manual_check_n"]), seed)
    write_csv(out / "manual_check.csv",
              ["qid", "relation", "relation_group", "subject", "wiki_url", "question", "pred", "gold",
               "label_m3", "cooc_page", "cooc_count", "disambig_exists", "manual_label", "note"],
              [[r["qid"], r["relation"], r["relation_group"], r["subject"],
                WIKI_URL + str(r.get("wiki_canonical") or "").replace(" ", "_"), r["question"], r["pred"],
                " | ".join(r["aliases"][:3]), r["label_m3"], r["cooc_page"], r["cooc_count"], r["disambig_exists"], "", ""]
               for r in sample])
    write_csv(out / "sentence_answers.csv", ["qid", "relation", "subject", "pred"],
              [[r["qid"], r["relation"], r["subject"], r["pred"]] for r in rows if r.get("m3_status") == "sentence"])
    write_json(out / "summary.json", s)
    write_meta(out, cfg, "labeling", {"summary": {k: s[k] for k in ("n", "n_wrong_scored", "labels_by_rule")}})
    if report is not None:
        fill_report(report, s, lc, rows, mpath)
    print(f"[labeling] {s['labels_by_rule'][s['primary_rule']]} -> {out}")
