"""Stage `correctness` (M1.6): runs on saved generations, no GPU (CLAUDE.md §3 Correctness).

For the greedy answer and every sample:
  normalize -> drop generic aliases -> abstain (refusal patterns, per-relation exemptions)
  -> match_strict (incl. head-word-stripped aliases) -> correct
  -> match_relaxed only -> partial  -> otherwise wrong
Final label = override if present, else label_auto.
`partial`, `abstain` and `gold_noise` rows are excluded from FA/AH/UH training. The
`ambiguous_subject` heuristic is a column only and excludes nothing.

Outputs (outputs/<run>/correctness/):
  labels.jsonl      one row per question
  abstain_log.csv   every refusal match (greedy + samples)
  partial.csv       every partial greedy answer
  wrong.csv         every wrong greedy answer
  summary.json, meta.json
Labels are per model: never reuse them across models.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..reporting.console import bar, model_short, pct, table
from ..reporting.plots import outcomes_by_relation
from ..utils import (
    find_upstream, read_jsonl, report_dir, resolve_pkg_path, run_dir, write_json, write_jsonl,
    write_meta,
)
from .matching import LABELS, RefusalPatterns, auto_label, load_head_words, match
from .review import (
    ambiguous_subject_heuristic, clean_aliases, load_generic_aliases, load_overrides,
    load_relation_groups, write_csv,
)

DEFAULTS = {
    "refusal_patterns": "configs/refusal_patterns.yaml",
    "generic_aliases": "configs/generic_aliases.yaml",
    "head_words": "configs/head_words.yaml",
    "relation_groups": "configs/relation_groups.yaml",
}
N_PARTIAL_EXAMPLES = 10


def overrides_path(cfg: dict[str, Any]) -> Path:
    """overrides/<dataset>_<model>.csv unless the config names a file."""
    ccfg = cfg.get("correctness", {}) or {}
    rel = ccfg.get("overrides") or f"overrides/{cfg['data']['dataset']}_{model_short(cfg)}.csv"
    return resolve_pkg_path(rel)  # type: ignore[return-value]


# ----------------------------------------------------------------------------- per row

class Judge:
    """Holds the matching settings and config lists; judges one answer."""

    def __init__(self, ccfg: dict[str, Any]) -> None:
        def path(key: str) -> Path:
            return resolve_pkg_path(ccfg.get(key, DEFAULTS[key]))  # type: ignore[return-value]

        self.paths = {k: path(k) for k in DEFAULTS}
        self.refusal = RefusalPatterns.load(self.paths["refusal_patterns"])
        self.generic = load_generic_aliases(self.paths["generic_aliases"])
        self.head_words = load_head_words(self.paths["head_words"])
        self.groups, self.borderline = load_relation_groups(self.paths["relation_groups"])
        self.min_words = ccfg.get("relaxed_min_words", 2)
        self.min_chars = ccfg.get("relaxed_min_chars", 6)

    def __call__(self, text: str, aliases: list[str], relation: str) -> dict[str, Any]:
        m = match(text, aliases, self.min_words, self.min_chars, self.head_words)
        label, pattern = auto_label(m, self.refusal, relation)
        return {**m, "label_auto": label, "refusal_pattern": pattern}

    def group(self, relation: str) -> str:
        return self.groups.get(relation.strip().lower(), "unassigned")


def previous_label(prev: dict[str, Any] | None) -> str | None:
    """Label from an earlier labels.jsonl (M1 rows only had a `correct` bool)."""
    if prev is None:
        return None
    if "label" in prev:
        return prev["label"]
    if "correct" in prev:
        return "correct" if prev["correct"] else "wrong"
    return None


def label_question(
    rec: dict[str, Any], q: dict[str, Any], judge: Judge, ccfg: dict[str, Any],
    override: dict[str, Any], prev: dict[str, Any] | None, dataset: str,
) -> dict[str, Any]:
    meta = q.get("meta") or {}
    relation = meta.get("prop") or "unknown"
    aliases, dropped = clean_aliases(rec["aliases"], judge.generic)
    g = judge(rec["greedy"]["text"], aliases, relation)
    samples = [judge(s["text"], aliases, relation) for s in rec["samples"]]
    sample_labels = [s["label_auto"] for s in samples]
    n_s = len(samples)

    label = override.get("label", g["label_auto"])
    ambiguous = bool(override.get("ambiguous_subject", ambiguous_subject_heuristic(rec.get("subject"), ccfg)))
    gold_noise = bool(override.get("gold_noise", False))
    return {
        "qid": rec["qid"],
        "dataset": dataset,
        "split": rec["split"],
        "relation": relation,
        "relation_group": judge.group(relation),  # analysis only, never a feature
        "s_pop": meta.get("s_pop"),
        "subject": rec.get("subject"),
        "question": rec["question"],
        "pred": rec["greedy"]["text"],
        "aliases": aliases,
        "aliases_dropped": dropped,
        **{k: g[k] for k in ("pred_norm", "aliases_stripped", "match_strict", "match_relaxed",
                              "strict_via_head_word", "matched_alias_strict", "matched_alias_relaxed",
                              "refusal_pattern")},
        "label_auto": g["label_auto"],
        "label": label,
        "overridden": bool(override),
        "label_overridden": "label" in override and override["label"] != g["label_auto"],
        "label_prev": previous_label(prev),
        "flags": {"ambiguous_subject": ambiguous, "gold_noise": gold_noise},
        # Rule 9: abstain, partial and gold_noise rows never enter FA/AH/UH training.
        "exclude_from_training": label in ("abstain", "partial") or gold_noise,
        "note": override.get("note"),
        "sample_labels": sample_labels,
        "sample_refusal_patterns": [s["refusal_pattern"] for s in samples],
        "sample_accuracy": sample_labels.count("correct") / n_s if n_s else None,
        "abstain_frac": sample_labels.count("abstain") / n_s if n_s else None,  # training-time only
    }


# ----------------------------------------------------------------------------- summary

def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    return {"n": n, **{lab: c[lab] for lab in LABELS}, "acc": c["correct"] / n if n else 0.0}


def summarize(labels: list[dict[str, Any]], unknown_override_qids: list[str], judge: Judge, seed: int) -> dict[str, Any]:
    n = len(labels)
    auto = Counter(r["label_auto"] for r in labels)
    final = Counter(r["label"] for r in labels)
    have_prev = [r for r in labels if r["label_prev"] is not None]
    changes = Counter(
        f"{r['label_prev']} → {r['label']}" + (" (overrides)" if r["label_overridden"] else "")
        for r in have_prev if r["label_prev"] != r["label"]
    )
    by_rel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in labels:
        by_rel[r["relation"]].append(r)
        by_group[r["relation_group"]].append(r)
        by_split[r["split"]].append(r)
    relations = sorted(
        ({"relation": k, "group": v[0]["relation_group"], **_counts(v)} for k, v in by_rel.items()),
        key=lambda d: (-d["n"], d["relation"]),
    )
    groups = {g: _counts(v) for g, v in sorted(by_group.items())}
    borderline = {b: _counts(by_rel[k]) for k in by_rel for b in judge.borderline if k.lower() == b}

    partial_rows = [r for r in labels if r["label"] == "partial"]
    rng = random.Random(seed)
    examples = rng.sample(partial_rows, min(N_PARTIAL_EXAMPLES, len(partial_rows)))
    sample_labels = Counter(lab for r in labels for lab in r["sample_labels"])
    answered = n - final["abstain"]
    return {
        "n": n,
        "counts_auto": {lab: auto[lab] for lab in LABELS},
        "counts": {lab: final[lab] for lab in LABELS},
        "accuracy": final["correct"] / n if n else 0.0,
        "accuracy_answered": final["correct"] / answered if answered else 0.0,
        "abstain_rate": final["abstain"] / n if n else 0.0,
        "partial_rate": final["partial"] / n if n else 0.0,
        "accuracy_by_split": {k: _counts(v)["acc"] for k, v in sorted(by_split.items())},
        "previous_run": {"n_compared": len(have_prev), "changes": dict(changes.most_common())},
        "head_word_matches": sum(r["strict_via_head_word"] and r["label"] == "correct" for r in labels),
        "overrides": {
            "rows_applied": sum(r["overridden"] for r in labels),
            "label_changed": sum(r["label_overridden"] for r in labels),
            "unknown_qids": unknown_override_qids,
        },
        "flags": {
            "ambiguous_subject": sum(r["flags"]["ambiguous_subject"] for r in labels),
            "gold_noise": sum(r["flags"]["gold_noise"] for r in labels),
        },
        "excluded_from_training": sum(r["exclude_from_training"] for r in labels),
        "aliases_dropped": dict(Counter(a.lower() for r in labels for a in r["aliases_dropped"]).most_common()),
        "questions_with_dropped_alias": sum(bool(r["aliases_dropped"]) for r in labels),
        "refusal_patterns_greedy": dict(Counter(r["refusal_pattern"] for r in labels if r["refusal_pattern"]).most_common()),
        "refusal_patterns_samples": dict(Counter(p for r in labels for p in r["sample_refusal_patterns"] if p).most_common()),
        "samples": {"n": sum(sample_labels.values()), **{lab: sample_labels[lab] for lab in LABELS}},
        "by_relation": relations,
        "by_group": groups,
        "borderline": borderline,
        "unassigned_relations": sorted(r["relation"] for r in relations if r["group"] == "unassigned"),
        "partial_examples": [
            {"qid": r["qid"], "relation": r["relation"], "pred": r["pred"], "gold": r["matched_alias_relaxed"]}
            for r in examples
        ],
    }


def _fmt_counter(d: dict[str, int]) -> str:
    return ", ".join(f"{k} ×{v}" for k, v in d.items()) or "none"


def _label_row(name: str, c: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    n = c["n"]
    return [name, *(extra or []), str(n)] + [pct(c[lab], n) for lab in LABELS] + [bar(c["acc"], 10)]


def fill_report(part: Any, s: dict[str, Any], cfg: dict[str, Any], ov_path: Path) -> None:
    ccfg = cfg.get("correctness", {}) or {}
    n, c = s["n"], s["counts"]
    part.n = n
    part.summary = s

    # ---- checks
    max_abst = ccfg.get("max_abstain_rate", 0.15)
    part.soft(s["abstain_rate"] < max_abst, f"abstain rate {100 * s['abstain_rate']:.1f}%"
              + ("" if s["abstain_rate"] < max_abst else f" (expected < {100 * max_abst:.0f}%: check the refusal patterns)"))
    max_part = ccfg.get("max_partial_rate", 0.05)
    part.soft(s["partial_rate"] < max_part, f"partial rate {100 * s['partial_rate']:.1f}%"
              + ("" if s["partial_rate"] < max_part else f" (expected < {100 * max_part:.0f}%: relaxed matching may be too loose)"))
    lo, hi = ccfg.get("expected_accuracy", [0.10, 0.30])
    ok = lo <= s["accuracy"] <= hi
    part.soft(ok, f"accuracy {100 * s['accuracy']:.1f}%" + ("" if ok else f" (expected {100 * lo:.0f}–{100 * hi:.0f}% for this dataset)"))
    un = s["unassigned_relations"]
    part.soft(not un, "every relation is in an open/closed group" if not un
              else f"relations without a group in relation_groups.yaml: {', '.join(un)}")

    # ---- key numbers
    part.key.append(f"correct {pct(c['correct'], n)} | wrong {pct(c['wrong'], n)} | "
                    f"abstain {pct(c['abstain'], n)} | partial {pct(c['partial'], n)}")
    part.key.append(f"excluded from training {s['excluded_from_training']} (abstain + partial + gold_noise) | "
                    f"flags: ambiguous_subject {s['flags']['ambiguous_subject']} (column only), "
                    f"gold_noise {s['flags']['gold_noise']}")
    prev = s["previous_run"]
    if not prev["n_compared"]:
        ch = "no previous labels found"
    elif not prev["changes"]:
        ch = "none"
    else:
        ch = ", ".join(f"{v} {k}" for k, v in prev["changes"].items())
    part.key.append(f"changed vs previous run: {ch}")
    part.key.append(f"head-word matches: {s['head_word_matches']} | aliases dropped as generic: "
                    f"{sum(s['aliases_dropped'].values())}"
                    + (f" ({_fmt_counter(s['aliases_dropped'])})" if s["aliases_dropped"] else ""))
    part.key.append(f"overrides applied {s['overrides']['rows_applied']} | refusals: {_fmt_counter(s['refusal_patterns_greedy'])}")

    head = ["", "n", "correct", "wrong", "abstain", "partial", "acc"]
    part.key.append("by group")
    part.key += table([head] + [_label_row(g, gc) for g, gc in s["by_group"].items()]
                      + [_label_row(f"({b}: borderline)", bc) for b, bc in s["borderline"].items()], indent="  ")
    part.key.append("by relation")
    part.key += table([["", "group", *head[1:]]]
                      + [_label_row(r["relation"], r, [r["group"]]) for r in s["by_relation"]], indent="  ")
    if s["partial_examples"]:
        part.key.append(f"partial examples ({len(s['partial_examples'])} random of {c['partial']}): pred → gold")
        part.key += [f"  {e['qid']} [{e['relation']}] {e['pred']!r} → {e['gold']!r}" for e in s["partial_examples"]]

    # ---- for Claude
    cl = part.claude
    cl.append(f"counts auto={s['counts_auto']} final={c}")
    cl.append(f"acc={s['accuracy']:.4f} acc_answered={s['accuracy_answered']:.4f} by_split="
              + str({k: round(v, 4) for k, v in s["accuracy_by_split"].items()}))
    cl.append(f"changes_vs_prev={prev['changes']} (n_compared={prev['n_compared']})")
    cl.append(f"head_word_matches={s['head_word_matches']} aliases_dropped={s['aliases_dropped']} "
              f"in {s['questions_with_dropped_alias']} questions")
    cl.append(f"overrides file={ov_path.name} applied={s['overrides']['rows_applied']} "
              f"label_changed={s['overrides']['label_changed']} unknown_qids={len(s['overrides']['unknown_qids'])}")
    cl.append(f"flags={s['flags']} excluded_from_training={s['excluded_from_training']}")
    cl.append(f"refusals greedy={s['refusal_patterns_greedy']}")
    cl.append(f"refusals samples={s['refusal_patterns_samples']}")
    sm = s["samples"]
    cl.append("samples " + " ".join(f"{k}={v}" for k, v in sm.items()))
    cl.append("groups " + "; ".join(f"{g}: n={gc['n']} acc={gc['acc']:.3f} partial={gc['partial']}"
                                    for g, gc in s["by_group"].items()))
    cl.append("(relation table: see KEY NUMBERS)")

    part.plots.append(("Greedy outcomes by relation", outcomes_by_relation(s["by_relation"])))


# ----------------------------------------------------------------------------- stage

def _required(cfg: dict[str, Any], rel: str) -> Path:
    p = find_upstream(cfg, rel)
    if p is None:
        raise FileNotFoundError(
            f"{rel} not found under {run_dir(cfg)} or in the attached inputs "
            f"(input_run_dir={cfg.get('input_run_dir')!r}). Attach the Kaggle Dataset with the "
            f"saved outputs, or pass --input-run-dir."
        )
    return p


def run(cfg: dict[str, Any], report: Any = None) -> None:
    ccfg = cfg.get("correctness", {}) or {}
    dataset = cfg["data"]["dataset"]
    rd = run_dir(cfg)
    out = rd / "correctness"

    gen_path = _required(cfg, "generation/generations.jsonl")
    q_path = _required(cfg, "data/questions.jsonl")
    cfg["input_run_dir_used"] = str(gen_path.parent.parent)
    records = read_jsonl(gen_path)
    questions = {q["qid"]: q for q in read_jsonl(q_path)}
    prev_path = find_upstream(cfg, "correctness/labels.jsonl")
    prev = {r["qid"]: r for r in read_jsonl(prev_path)} if prev_path else {}

    judge = Judge(ccfg)
    ov_path = overrides_path(cfg)
    overrides = load_overrides(ov_path)
    unknown = sorted(set(overrides) - {r["qid"] for r in records})

    labels = [
        label_question(r, questions.get(r["qid"], {}), judge, ccfg,
                       overrides.get(r["qid"], {}), prev.get(r["qid"]), dataset)
        for r in records
    ]
    write_jsonl(out / "labels.jsonl", labels)

    log_rows = [[lab["qid"], "greedy", lab["pred"], lab["refusal_pattern"]] for lab in labels if lab["refusal_pattern"]]
    for rec, lab in zip(records, labels):
        for i, (smp, p) in enumerate(zip(rec["samples"], lab["sample_refusal_patterns"])):
            if p:
                log_rows.append([rec["qid"], f"sample_{i}", smp["text"], p])
    write_csv(out / "abstain_log.csv", ["qid", "which", "text", "pattern"], log_rows)
    cols = ["qid", "relation", "subject", "question", "pred", "aliases", "matched_alias_relaxed", "flag_ambiguous"]
    for name in ("partial", "wrong"):
        write_csv(out / f"{name}.csv", cols, [
            [lab["qid"], lab["relation"], lab["subject"], lab["question"], lab["pred"],
             " | ".join(lab["aliases"]), lab["matched_alias_relaxed"], int(lab["flags"]["ambiguous_subject"])]
            for lab in labels if lab["label"] == name
        ])
    stale = report_dir(cfg) / "REVIEW.csv"  # no manual review list by default (§3 step 9)
    if stale.exists():
        stale.unlink()

    summary = summarize(labels, unknown, judge, cfg.get("seed", 0))
    summary["inputs"] = {
        "generations": str(gen_path), "questions": str(q_path),
        "previous_labels": str(prev_path) if prev_path else None,
        **{k: str(v) for k, v in judge.paths.items()},
        "overrides": str(ov_path), "overrides_found": ov_path.exists(),
    }
    write_json(out / "summary.json", summary)
    write_meta(out, cfg, "correctness", {"summary": {k: summary[k] for k in ("n", "counts", "accuracy", "flags")}})
    if report is not None:
        fill_report(report, summary, cfg, ov_path)
    print(f"[correctness] n={summary['n']} acc={100 * summary['accuracy']:.1f}% "
          f"partial={summary['counts']['partial']} -> {out}")
