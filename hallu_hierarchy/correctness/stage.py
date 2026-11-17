"""Stage `correctness` (M1.5): runs on saved generations, no GPU.

For the greedy answer and every sample:
  normalize -> abstain (refusal patterns) -> match_strict -> label; match_relaxed for review only.
Then flags (ambiguous_subject heuristic, gold_noise manual), manual overrides, review list,
comparison with the previous run's labels, console summary block, summary.json, plots.

Outputs in <run>/correctness/:
  labels.jsonl                 one row per question (label_auto, label, label_prev, flags, ...)
  review/disagreements.csv     strict != relaxed (greedy)
  review/wrong.csv             all wrong answers (for spotting gold noise)
  abstain_log.csv              every refusal-pattern match (greedy and samples), for auditing
  summary.json, meta.json
Labels are per model: never reuse them across models.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..reporting.console import bar, block, kv, pct
from ..reporting.plots import outcomes_by_relation
from ..utils import (
    find_upstream, read_jsonl, resolve_pkg_path, run_dir, write_json, write_jsonl, write_meta,
)
from .matching import LABELS, RefusalPatterns, auto_label, match, relaxed_label
from .review import (
    ambiguous_subject_heuristic, load_overrides, write_csv, write_review_csv,
)

DEFAULT_REFUSALS = "configs/refusal_patterns.yaml"


# ----------------------------------------------------------------------------- per row

def judge(text: str, aliases: list[str], refusal: RefusalPatterns, ccfg: dict[str, Any]) -> dict[str, Any]:
    m = match(text, aliases, ccfg.get("relaxed_min_words", 2), ccfg.get("relaxed_min_chars", 6))
    label, pattern = auto_label(m, refusal)
    return {**m, "label_auto": label, "label_relaxed": relaxed_label(m, label), "refusal_pattern": pattern}


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
    rec: dict[str, Any], q: dict[str, Any], refusal: RefusalPatterns, ccfg: dict[str, Any],
    override: dict[str, Any], prev: dict[str, Any] | None,
) -> dict[str, Any]:
    g = judge(rec["greedy"]["text"], rec["aliases"], refusal, ccfg)
    samples = [judge(s["text"], rec["aliases"], refusal, ccfg) for s in rec["samples"]]
    sample_labels = [s["label_auto"] for s in samples]
    n_s = len(samples)

    ambiguous = override.get("ambiguous_subject", ambiguous_subject_heuristic(rec.get("subject"), ccfg))
    gold_noise = override.get("gold_noise", False)
    label = override.get("decision", g["label_auto"])
    meta = q.get("meta") or {}
    return {
        "qid": rec["qid"],
        "split": rec["split"],
        "relation": meta.get("prop") or "unknown",
        "s_pop": meta.get("s_pop"),
        "subject": rec.get("subject"),
        "question": rec["question"],
        "pred": rec["greedy"]["text"],
        "aliases": rec["aliases"],
        **{k: g[k] for k in ("pred_norm", "aliases_norm", "match_strict", "match_relaxed",
                              "matched_alias_strict", "matched_alias_relaxed", "refusal_pattern")},
        "label_auto": g["label_auto"],
        "label_relaxed": g["label_relaxed"],
        "label": label,
        "label_overridden": "decision" in override and override["decision"] != g["label_auto"],
        "label_prev": previous_label(prev),
        "flags": {"ambiguous_subject": bool(ambiguous), "gold_noise": bool(gold_noise)},
        # flagged or abstaining rows never enter FA/AH/UH training
        "exclude_from_training": bool(ambiguous or gold_noise or label == "abstain"),
        "review_note": override.get("note"),
        "sample_labels": sample_labels,
        "sample_match_relaxed": [s["match_relaxed"] for s in samples],
        "sample_refusal_patterns": [s["refusal_pattern"] for s in samples],
        "sample_accuracy": sample_labels.count("correct") / n_s if n_s else None,
        "abstain_frac": sample_labels.count("abstain") / n_s if n_s else None,  # training-time only
    }


# ----------------------------------------------------------------------------- summary

def summarize(labels: list[dict[str, Any]], overrides_unused: list[str]) -> dict[str, Any]:
    n = len(labels)
    counts = {
        col: {lab: sum(r[key] == lab for r in labels) for lab in LABELS}
        for col, key in (("strict", "label_auto"), ("relaxed", "label_relaxed"), ("final", "label"))
    }
    have_prev = [r for r in labels if r["label_prev"] is not None]
    changes = Counter(f"{r['label_prev']} → {r['label']}" for r in have_prev if r["label_prev"] != r["label"])

    rel: dict[str, Counter] = defaultdict(Counter)
    for r in labels:
        rel[r["relation"]][r["label"]] += 1
    by_relation = sorted(
        (
            {"relation": k, "n": sum(c.values()), **{lab: c[lab] for lab in LABELS},
             "acc": c["correct"] / sum(c.values())}
            for k, c in rel.items()
        ),
        key=lambda d: (-d["n"], d["relation"]),
    )
    by_split: dict[str, list[str]] = defaultdict(list)
    for r in labels:
        by_split[r["split"]].append(r["label"])

    sample_labels = [lab for r in labels for lab in r["sample_labels"]]
    n_s = len(sample_labels)
    patterns = Counter(r["refusal_pattern"] for r in labels if r["refusal_pattern"])
    sample_patterns = Counter(p for r in labels for p in r["sample_refusal_patterns"] if p)
    final = counts["final"]
    return {
        "n": n,
        "counts": counts,
        "accuracy": final["correct"] / n if n else None,
        "accuracy_answered": final["correct"] / (n - final["abstain"]) if n - final["abstain"] else None,
        "accuracy_by_split": {k: v.count("correct") / len(v) for k, v in sorted(by_split.items())},
        "previous_run": {"n_compared": len(have_prev), "changes": dict(changes.most_common())},
        "review_list": sum(r["match_strict"] != r["match_relaxed"] and r["label_auto"] != "abstain" for r in labels),
        "overrides": {
            "label_changed": sum(r["label_overridden"] for r in labels),
            "unknown_qids": overrides_unused,
        },
        "flags": {
            "ambiguous_subject": sum(r["flags"]["ambiguous_subject"] for r in labels),
            "gold_noise": sum(r["flags"]["gold_noise"] for r in labels),
        },
        "excluded_from_training": sum(r["exclude_from_training"] for r in labels),
        "refusal_patterns_greedy": dict(patterns.most_common()),
        "refusal_patterns_samples": dict(sample_patterns.most_common()),
        "samples": {
            "n": n_s,
            **{lab: sample_labels.count(lab) for lab in LABELS},
            "mean_sample_accuracy": (
                sum(r["sample_accuracy"] for r in labels if r["sample_accuracy"] is not None) / n if n else None
            ),
        },
        "accuracy_by_relation": by_relation,
    }


def render(s: dict[str, Any], cfg: dict[str, Any], review_rel: str) -> str:
    n, c = s["n"], s["counts"]
    lines = [f"{'Outcome counts':<20}{'strict':>14}{'relaxed':>14}{'final':>14}"]
    for lab in LABELS:
        lines.append(f"  {lab:<18}" + "".join(f"{pct(c[col][lab], n):>14}" for col in ("strict", "relaxed", "final")))
    lines.append("  (relaxed is for review only; final = strict + manual overrides)")
    lines.append(kv("Accuracy", f"{pct(c['final']['correct'], n).strip()} of all | "
                    f"{100 * (s['accuracy_answered'] or 0):.1f}% of answered"))
    split_acc = " | ".join(f"{k} {100 * v:.1f}%" for k, v in s["accuracy_by_split"].items())
    lines.append(kv("Accuracy by split", split_acc))

    prev = s["previous_run"]
    if not prev["n_compared"]:
        lines.append(kv("Changed vs previous run:", "no previous labels found"))
    elif not prev["changes"]:
        lines.append(kv("Changed vs previous run:", f"no changes (n={prev['n_compared']})"))
    else:
        items = [f"{v} {k}" for k, v in prev["changes"].items()]
        lines.append(kv("Changed vs previous run:", items[0]))
        lines += [kv("", it) for it in items[1:]]

    lines.append(kv("Review list:", f"{s['review_list']} rows   → {review_rel}"))
    ov = s["overrides"]
    ov_txt = f"{ov['label_changed']} labels changed"
    if ov["unknown_qids"]:
        ov_txt += f" | {len(ov['unknown_qids'])} qids not in this run (ignored)"
    lines.append(kv("Manual overrides:", ov_txt))
    f = s["flags"]
    lines.append(kv("Flags:", f"ambiguous_subject {pct(f['ambiguous_subject'], n).strip()} | "
                    f"gold_noise {pct(f['gold_noise'], n).strip()}"))
    lines.append(kv("Excluded from training:", f"{pct(s['excluded_from_training'], n).strip()}  (abstain or flagged)"))
    pats = ", ".join(f"{k} ×{v}" for k, v in s["refusal_patterns_greedy"].items()) or "none"
    lines.append(kv("Refusals (greedy):", pats))

    sm = s["samples"]
    if sm["n"]:
        lines.append(kv("Samples:", f"n={sm['n']}  correct {pct(sm['correct'], sm['n']).strip()} | "
                        f"wrong {pct(sm['wrong'], sm['n']).strip()} | abstain {pct(sm['abstain'], sm['n']).strip()}"))

    rows = s["accuracy_by_relation"]
    width = max((len(r["relation"]) for r in rows), default=8)
    lines.append(f"Accuracy by relation ({len(rows)} relations, by n)")
    for r in rows:
        lines.append(f"  {r['relation']:<{width}}  n={r['n']:<4} acc {100 * r['acc']:5.1f}%  "
                     f"abst {r['abstain']:<3} {bar(r['acc'])}")
    return block("correctness", cfg, n, lines)


# ----------------------------------------------------------------------------- stage

def _review_row(lab: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return {
        **lab,
        "aliases": " | ".join(lab["aliases"]),
        "flag_ambiguous_subject": int(lab["flags"]["ambiguous_subject"]),
        "decision": override.get("decision", ""),
        "gold_noise": "" if "gold_noise" not in override else int(override["gold_noise"]),
        "ambiguous_subject": "" if "ambiguous_subject" not in override else int(override["ambiguous_subject"]),
        "note": override.get("note", ""),
    }


def _required(cfg: dict[str, Any], rel: str) -> Path:
    p = find_upstream(cfg, rel)
    if p is None:
        raise FileNotFoundError(
            f"{rel} not found under {run_dir(cfg)} or input_run_dir={cfg.get('input_run_dir')!r}. "
            "Pass --input-run-dir pointing at saved outputs (e.g. a Kaggle Dataset)."
        )
    return p


def run(cfg: dict[str, Any]) -> None:
    ccfg = cfg.get("correctness", {}) or {}
    rd = run_dir(cfg)
    out = rd / "correctness"

    gen_path = _required(cfg, "generation/generations.jsonl")
    q_path = _required(cfg, "data/questions.jsonl")
    records = read_jsonl(gen_path)
    questions = {q["qid"]: q for q in read_jsonl(q_path)}
    prev_path = find_upstream(cfg, "correctness/labels.jsonl")
    prev = {r["qid"]: r for r in read_jsonl(prev_path)} if prev_path else {}

    refusal_path = resolve_pkg_path(ccfg.get("refusal_patterns", DEFAULT_REFUSALS))
    refusal = RefusalPatterns.load(refusal_path)
    overrides_path = resolve_pkg_path(ccfg.get("overrides"))
    overrides = load_overrides(overrides_path)
    qids = {r["qid"] for r in records}
    unused = sorted(set(overrides) - qids)

    labels = [
        label_question(r, questions.get(r["qid"], {}), refusal, ccfg, overrides.get(r["qid"], {}), prev.get(r["qid"]))
        for r in records
    ]
    write_jsonl(out / "labels.jsonl", labels)

    review = out / "review"
    disagree = [lab for lab in labels if lab["match_strict"] != lab["match_relaxed"] and lab["label_auto"] != "abstain"]
    write_review_csv(review / "disagreements.csv", [_review_row(lab, overrides.get(lab["qid"], {})) for lab in disagree])
    wrong = [lab for lab in labels if lab["label"] == "wrong"]
    write_review_csv(review / "wrong.csv", [_review_row(lab, overrides.get(lab["qid"], {})) for lab in wrong])

    log_rows = []
    for lab in labels:
        if lab["refusal_pattern"]:
            log_rows.append([lab["qid"], "greedy", lab["pred"], lab["refusal_pattern"]])
    for rec, lab in zip(records, labels):
        for i, (s, p) in enumerate(zip(rec["samples"], lab["sample_refusal_patterns"])):
            if p:
                log_rows.append([rec["qid"], f"sample_{i}", s["text"], p])
    write_csv(out / "abstain_log.csv", ["qid", "which", "text", "pattern"], log_rows)

    summary = summarize(labels, unused)
    summary["inputs"] = {
        "generations": str(gen_path), "questions": str(q_path),
        "previous_labels": str(prev_path) if prev_path else None,
        "refusal_patterns": str(refusal_path), "overrides": str(overrides_path) if overrides_path else None,
        "overrides_found": bool(overrides_path and overrides_path.exists()),
    }
    plot = outcomes_by_relation(
        summary["accuracy_by_relation"], rd / "plots" / "correctness_by_relation.png",
        f"Greedy outcomes by relation (n={summary['n']})",
    )
    summary["plots"] = [str(plot)] if plot else []
    write_json(out / "summary.json", summary)
    write_meta(out, cfg, "correctness", {"summary": {k: summary[k] for k in ("n", "counts", "accuracy", "flags")}})
    write_meta(review, cfg, "correctness/review")

    print(render(summary, cfg, "review/disagreements.csv"))
    print(f"[correctness] outputs -> {out}" + (f" | plot -> {plot}" if plot else " | (matplotlib missing: no plot)"))
