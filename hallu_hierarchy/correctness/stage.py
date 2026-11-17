"""Stage `correctness`: runs on saved generations, no GPU (CLAUDE.md §3 Correctness).

For the greedy answer and every sample:
  normalize -> drop generic aliases -> abstain (refusal patterns) -> match_strict -> label_auto
  match_relaxed is for review only. Final label = override if present, else label_auto.
Flags: ambiguous_subject (heuristic, override can set/clear), gold_noise (override only).

Outputs:
  outputs/<run>/correctness/labels.jsonl   one row per question
  outputs/<run>/correctness/abstain_log.csv every refusal match (greedy + samples)
  outputs/<run>/correctness/wrong.csv      all wrong answers (for spotting gold noise)
  outputs/<run>/correctness/summary.json, meta.json
  REVIEW.csv (report folder)               strict != relaxed rows without an override
Labels are per model: never reuse them across models.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..reporting.console import bar, model_short, pct, table
from ..reporting.plots import outcomes_by_relation
from ..utils import (
    find_upstream, read_jsonl, report_dir, resolve_pkg_path, run_dir, write_json, write_jsonl,
    write_meta,
)
from .matching import LABELS, RefusalPatterns, auto_label, match, relaxed_label
from .review import (
    ambiguous_subject_heuristic, clean_aliases, load_generic_aliases, load_overrides, write_csv,
)

DEFAULT_REFUSALS = "configs/refusal_patterns.yaml"
DEFAULT_GENERIC = "configs/generic_aliases.yaml"
MAX_ACTION_ROWS = 20
REVIEW_HEADER = ["qid", "relation", "question", "pred", "aliases", "strict", "relaxed",
                 "matched_alias_relaxed", "label_auto"]


def overrides_path(cfg: dict[str, Any]) -> Path:
    """overrides/<dataset>_<model>.csv unless the config names a file."""
    ccfg = cfg.get("correctness", {}) or {}
    rel = ccfg.get("overrides") or f"overrides/{cfg['data']['dataset']}_{model_short(cfg)}.csv"
    return resolve_pkg_path(rel)  # type: ignore[return-value]


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
    rec: dict[str, Any], q: dict[str, Any], refusal: RefusalPatterns, generic: set[str],
    ccfg: dict[str, Any], override: dict[str, Any], prev: dict[str, Any] | None, dataset: str,
) -> dict[str, Any]:
    aliases, dropped = clean_aliases(rec["aliases"], generic)
    g = judge(rec["greedy"]["text"], aliases, refusal, ccfg)
    samples = [judge(s["text"], aliases, refusal, ccfg) for s in rec["samples"]]
    sample_labels = [s["label_auto"] for s in samples]
    n_s = len(samples)

    label = override.get("label", g["label_auto"])
    ambiguous = bool(override.get("ambiguous_subject", ambiguous_subject_heuristic(rec.get("subject"), ccfg)))
    gold_noise = bool(override.get("gold_noise", False))
    meta = q.get("meta") or {}
    return {
        "qid": rec["qid"],
        "dataset": dataset,
        "split": rec["split"],
        "relation": meta.get("prop") or "unknown",
        "s_pop": meta.get("s_pop"),
        "subject": rec.get("subject"),
        "question": rec["question"],
        "pred": rec["greedy"]["text"],
        "aliases": aliases,
        "aliases_dropped": dropped,
        **{k: g[k] for k in ("pred_norm", "match_strict", "match_relaxed",
                              "matched_alias_strict", "matched_alias_relaxed", "refusal_pattern")},
        "label_auto": g["label_auto"],
        "label_relaxed": g["label_relaxed"],
        "label": label,
        "overridden": bool(override),
        "label_overridden": "label" in override and override["label"] != g["label_auto"],
        "label_prev": previous_label(prev),
        "flags": {"ambiguous_subject": ambiguous, "gold_noise": gold_noise},
        # Rule 9: flagged rows never enter AH/UH training; gold-noise rows have no reliable label.
        "exclude_from_training": label == "abstain" or gold_noise or (ambiguous and label == "wrong"),
        "note": override.get("note"),
        "sample_labels": sample_labels,
        "sample_match_relaxed": [s["match_relaxed"] for s in samples],
        "sample_refusal_patterns": [s["refusal_pattern"] for s in samples],
        "sample_accuracy": sample_labels.count("correct") / n_s if n_s else None,
        "abstain_frac": sample_labels.count("abstain") / n_s if n_s else None,  # training-time only
    }


def needs_review(lab: dict[str, Any]) -> bool:
    return lab["match_strict"] != lab["match_relaxed"] and lab["label_auto"] != "abstain" and not lab["overridden"]


# ----------------------------------------------------------------------------- summary

def summarize(labels: list[dict[str, Any]], unknown_override_qids: list[str]) -> dict[str, Any]:
    n = len(labels)
    counts = {
        col: {lab: sum(r[key] == lab for r in labels) for lab in LABELS}
        for col, key in (("strict", "label_auto"), ("relaxed", "label_relaxed"), ("final", "label"))
    }
    have_prev = [r for r in labels if r["label_prev"] is not None]
    changes = Counter(
        f"{r['label_prev']} → {r['label']}" + (" (overrides)" if r["label_overridden"] else "")
        for r in have_prev if r["label_prev"] != r["label"]
    )
    rel: dict[str, Counter] = defaultdict(Counter)
    for r in labels:
        rel[r["relation"]][r["label"]] += 1
    by_relation = sorted(
        ({"relation": k, "n": sum(c.values()), **{lab: c[lab] for lab in LABELS},
          "acc": c["correct"] / sum(c.values())} for k, c in rel.items()),
        key=lambda d: (-d["n"], d["relation"]),
    )
    by_split: dict[str, list[str]] = defaultdict(list)
    for r in labels:
        by_split[r["split"]].append(r["label"])
    sample_labels = [lab for r in labels for lab in r["sample_labels"]]
    final = counts["final"]
    answered = n - final["abstain"]
    return {
        "n": n,
        "counts": counts,
        "accuracy": final["correct"] / n if n else 0.0,
        "accuracy_answered": final["correct"] / answered if answered else 0.0,
        "abstain_rate": final["abstain"] / n if n else 0.0,
        "accuracy_by_split": {k: v.count("correct") / len(v) for k, v in sorted(by_split.items())},
        "previous_run": {"n_compared": len(have_prev), "changes": dict(changes.most_common())},
        "review_rows": sum(needs_review(r) for r in labels),
        "overrides": {
            "rows_applied": sum(r["overridden"] for r in labels),
            "label_changed": sum(r["label_overridden"] for r in labels),
            "unknown_qids": unknown_override_qids,
        },
        "flags": {
            "ambiguous_subject": sum(r["flags"]["ambiguous_subject"] for r in labels),
            "gold_noise": sum(r["flags"]["gold_noise"] for r in labels),
        },
        "excluded_non_abstain": sum(r["exclude_from_training"] and r["label"] != "abstain" for r in labels),
        "aliases_dropped": dict(Counter(a.lower() for r in labels for a in r["aliases_dropped"]).most_common()),
        "questions_with_dropped_alias": sum(bool(r["aliases_dropped"]) for r in labels),
        "refusal_patterns_greedy": dict(Counter(r["refusal_pattern"] for r in labels if r["refusal_pattern"]).most_common()),
        "refusal_patterns_samples": dict(Counter(p for r in labels for p in r["sample_refusal_patterns"] if p).most_common()),
        "samples": {
            "n": len(sample_labels),
            **{lab: sample_labels.count(lab) for lab in LABELS},
        },
        "accuracy_by_relation": by_relation,
    }


def _fmt_counter(d: dict[str, int]) -> str:
    return ", ".join(f"{k} ×{v}" for k, v in d.items()) or "none"


def fill_report(part: Any, s: dict[str, Any], labels: list[dict[str, Any]], cfg: dict[str, Any],
                review_rel: str, ov_path: Path) -> None:
    ccfg = cfg.get("correctness", {}) or {}
    n, c = s["n"], s["counts"]
    part.n = n
    part.summary = s

    # ---- checks
    max_abst = ccfg.get("max_abstain_rate", 0.15)
    part.soft(s["abstain_rate"] < max_abst, f"abstain rate {100 * s['abstain_rate']:.1f}%"
              + ("" if s["abstain_rate"] < max_abst else f" (expected < {100 * max_abst:.0f}%: check the refusal patterns)"))
    lo, hi = ccfg.get("expected_accuracy", [0.10, 0.30])
    ok = lo <= s["accuracy"] <= hi
    part.soft(ok, f"accuracy {100 * s['accuracy']:.1f}%" + ("" if ok else f" (expected {100 * lo:.0f}–{100 * hi:.0f}% for this dataset)"))
    max_amb = ccfg.get("max_ambiguous_frac", 0.30)
    amb = s["flags"]["ambiguous_subject"] / n if n else 0.0
    part.soft(amb <= max_amb, f"ambiguous_subject flags {100 * amb:.1f}% of questions"
              + ("" if amb <= max_amb else f" (expected ≤ {100 * max_amb:.0f}%: the single-word heuristic flags too many; "
                 f"wrong answers among them are excluded from AH/UH training)"))
    part.soft(s["review_rows"] == 0, "no strict/relaxed disagreements left to decide" if not s["review_rows"]
              else f"{s['review_rows']} strict/relaxed disagreements need a decision (see ACTION NEEDED)")
    if s["overrides"]["unknown_qids"] and ccfg.get("warn_unknown_override_qids", False):
        part.soft(False, f"{len(s['overrides']['unknown_qids'])} override qids are not in this run")

    # ---- key numbers
    f = c["final"]
    excl = s["excluded_non_abstain"]
    part.key.append(f"correct {pct(f['correct'], n)} | wrong {pct(f['wrong'], n)} | "
                    f"abstain {pct(f['abstain'], n)} | excluded {excl}")
    part.key += table([["outcome", "strict", "relaxed", "final"]] + [
        [lab] + [pct(c[col][lab], n) for col in ("strict", "relaxed", "final")] for lab in LABELS
    ], indent="  ")
    prev = s["previous_run"]
    if not prev["n_compared"]:
        ch = "no previous labels found"
    elif not prev["changes"]:
        ch = "none"
    else:
        ch = ", ".join(f"{v} {k}" for k, v in prev["changes"].items())
    part.key.append(f"changed vs previous run: {ch}")
    part.key.append(f"aliases dropped as generic: {sum(s['aliases_dropped'].values())}"
                    + (f" ({_fmt_counter(s['aliases_dropped'])})" if s["aliases_dropped"] else ""))
    part.key.append(f"review rows {s['review_rows']} | overrides applied {s['overrides']['rows_applied']} | "
                    f"flags: ambiguous_subject {s['flags']['ambiguous_subject']}, gold_noise {s['flags']['gold_noise']}")
    rows = s["accuracy_by_relation"]
    w = max((len(r["relation"]) for r in rows[:5]), default=8)
    part.key.append("accuracy by relation (top 5 by n)")
    part.key += [f"  {r['relation']:<{w}}  n={r['n']:<5} {100 * r['acc']:5.1f}%  {bar(r['acc'])}" for r in rows[:5]]

    # ---- action needed
    todo = [lab for lab in labels if needs_review(lab)]
    if todo:
        part.action.append(f"{len(todo)} rows need a decision (strict and relaxed disagree)"
                           + (f"; first {MAX_ACTION_ROWS} below" if len(todo) > MAX_ACTION_ROWS else "")
                           + f" — also in {review_rel}")
        part.action.append("Tell Claude Code e.g. \"popqa-123 = correct, popqa-456 = wrong\".")
        part.action.append("qid | question | pred | aliases | strict | relaxed")
        for lab in todo[:MAX_ACTION_ROWS]:
            part.action.append(f"{lab['qid']} | {lab['question']} | {lab['pred']} | "
                               f"{' / '.join(lab['aliases'][:4])} | {int(lab['match_strict'])} | {int(lab['match_relaxed'])}")

    # ---- for Claude
    cl = part.claude
    cl.append(f"counts strict={c['strict']} relaxed={c['relaxed']} final={c['final']}")
    cl.append(f"acc={s['accuracy']:.4f} acc_answered={s['accuracy_answered']:.4f} by_split="
              + str({k: round(v, 4) for k, v in s["accuracy_by_split"].items()}))
    cl.append(f"changes_vs_prev={prev['changes']} (n_compared={prev['n_compared']})")
    cl.append(f"overrides file={ov_path.name} applied={s['overrides']['rows_applied']} "
              f"label_changed={s['overrides']['label_changed']} unknown_qids={len(s['overrides']['unknown_qids'])}")
    cl.append(f"flags={s['flags']} excluded_non_abstain={excl} review_rows={s['review_rows']}")
    cl.append(f"aliases_dropped={s['aliases_dropped']} in {s['questions_with_dropped_alias']} questions")
    cl.append(f"refusals greedy={s['refusal_patterns_greedy']}")
    cl.append(f"refusals samples={s['refusal_patterns_samples']}")
    sm = s["samples"]
    cl.append(f"samples n={sm['n']} correct={sm['correct']} wrong={sm['wrong']} abstain={sm['abstain']}")
    cl.append("by relation (relation n acc abstain):")
    cl += [f"  {r['relation']} {r['n']} {r['acc']:.3f} {r['abstain']}" for r in rows]

    part.plots.append(("Greedy outcomes by relation", outcomes_by_relation(rows)))


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

    refusal_path = resolve_pkg_path(ccfg.get("refusal_patterns", DEFAULT_REFUSALS))
    refusal = RefusalPatterns.load(refusal_path)
    generic_path = resolve_pkg_path(ccfg.get("generic_aliases", DEFAULT_GENERIC))
    generic = load_generic_aliases(generic_path)
    ov_path = overrides_path(cfg)
    overrides = load_overrides(ov_path)
    unknown = sorted(set(overrides) - {r["qid"] for r in records})

    labels = [
        label_question(r, questions.get(r["qid"], {}), refusal, generic, ccfg,
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
    write_csv(out / "wrong.csv", ["qid", "relation", "subject", "question", "pred", "aliases", "flag_ambiguous"],
              [[lab["qid"], lab["relation"], lab["subject"], lab["question"], lab["pred"],
                " | ".join(lab["aliases"]), int(lab["flags"]["ambiguous_subject"])]
               for lab in labels if lab["label"] == "wrong"])

    # REVIEW.csv only when action is needed; otherwise remove a stale one.
    review = report_dir(cfg) / "REVIEW.csv"
    todo = [lab for lab in labels if needs_review(lab)]
    if todo:
        write_csv(review, REVIEW_HEADER, [
            [lab["qid"], lab["relation"], lab["question"], lab["pred"], " | ".join(lab["aliases"]),
             int(lab["match_strict"]), int(lab["match_relaxed"]), lab["matched_alias_relaxed"], lab["label_auto"]]
            for lab in todo
        ])
    elif review.exists():
        review.unlink()

    summary = summarize(labels, unknown)
    summary["inputs"] = {
        "generations": str(gen_path), "questions": str(q_path),
        "previous_labels": str(prev_path) if prev_path else None,
        "refusal_patterns": str(refusal_path), "generic_aliases": str(generic_path),
        "overrides": str(ov_path), "overrides_found": ov_path.exists(),
    }
    write_json(out / "summary.json", summary)
    write_meta(out, cfg, "correctness", {"summary": {k: summary[k] for k in ("n", "counts", "accuracy", "flags")}})
    if report is not None:
        fill_report(report, summary, labels, cfg, review.name, ov_path)
    print(f"[correctness] n={summary['n']} acc={100 * summary['accuracy']:.1f}% "
          f"review_rows={summary['review_rows']} -> {out}")
