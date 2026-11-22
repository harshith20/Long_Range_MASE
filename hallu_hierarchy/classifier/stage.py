"""Stage `classifier` (M5): boundary classifiers on single-pass features (CLAUDE.md §3 Classifier, Baselines,
Metrics). CPU only, seconds to minutes.

Rows: M3 labels FA / AH / UH (ambiguous_confirmed, unscored, abstain, partial and gold_noise are left out).
Tasks (positive class second):
  hall      FA vs AH∪UH   stage 1 of the two-stage classifier (H1)
  ah_uh     AH vs UH      stage 2 (positive = UH, as in E16)
  fa_ah     FA vs AH      H2 / H3
  flat      FA / AH / UH  one multinomial model; compared with the two-stage composition
Feature sets (inference-time, single pass; the probe scores are the M4 out-of-fold scores):
  E   tok_mean, tok_max, probe_sep_sem_entropy, probe_sep_seq_nll, ratio probe_sem / (tok_mean + eps)
  L   + logit-lens entropies and l*        A  + attention to the subject
  F   + familiarity probe                  C  + direct correctness probe (reported separately: it is a baseline)
Models: logistic regression (median imputation, standardization, balanced weights, C chosen on an inner
subject split of the training folds) and gradient boosting (sklearn HistGradientBoosting, fixed settings,
balanced weights). Same 5 subject-grouped outer folds as every earlier stage.
Baselines: sampled semantic entropy, SEP alone, token entropy alone, correctness probe alone, relation-only
classifier (one-hot relation; anything near it detects question type, not hallucination).
Metrics: AUROC with subject-bootstrap 95% CI, AUPRC, ECE (LR), within-relation macro, open / closed,
leave-one-relation-out, sensitivity without the heuristic ambiguous_subject flag, inference time.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..eval.metrics import auroc, bootstrap_auroc_ci
from ..probes.core import inner_split
from ..reporting.console import table
from ..utils import find_upstream, read_jsonl, run_dir, write_json, write_jsonl, write_meta

DEFAULTS: dict[str, Any] = {
    "C_grid": [0.01, 0.1, 1.0, 10.0],
    "inner_frac": 0.2,
    "gb": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    "min_per_class": 30,
    "n_boot": 1000,
    "ratio_eps": 1e-3,
    "max_within_gap": 0.10,
    "main_set": "E+L+A+F",
}
LENS = ["lens_H_18", "lens_H_24", "lens_H_30", "lens_H_36", "lens_lstar"]
ATTN = ["attn_mean_18", "attn_max_18", "attn_mean_26", "attn_max_26", "attn_mean_32", "attn_max_32"]
BLOCKS = {
    "E": ["tok_mean", "tok_max", "probe_sep_sem_entropy", "probe_sep_seq_nll", "ratio_probe_sem_tok"],
    "L": LENS, "A": ATTN, "F": ["probe_familiarity"], "C": ["probe_correctness"],
}
SETS = ["E", "E+L", "E+L+A", "E+L+A+F", "E+L+A+F+C"]
TASKS = {"hall": (["FA"], ["AH", "UH"]), "ah_uh": (["AH"], ["UH"]), "fa_ah": (["FA"], ["AH"])}
BASELINES = {  # name -> (column, sign): score = sign * column, higher = positive class
    "sampled sem_entropy": ("sem_entropy", 1), "SEP alone": ("probe_sep_sem_entropy", 1),
    "token entropy (tok_max)": ("tok_max", 1), "correctness probe": ("probe_correctness", 1),
    "familiarity (flipped)": ("probe_familiarity", -1),
}


def _cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULTS, **(cfg.get("classifier") or {})}


def columns(s: str) -> list[str]:
    return [c for b in s.split("+") for c in BLOCKS[b]]


# ----------------------------------------------------------------------------- inputs

def load_rows(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {k: find_upstream(cfg, p) for k, p in (("m3", "labeling/labels_m3.jsonl"),
                                                   ("entropy", "entropy/features.jsonl"),
                                                   ("probes", "probes/oof_scores.jsonl"))}
    miss = [k for k, p in paths.items() if p is None]
    if miss:
        raise FileNotFoundError(f"classifier needs labeling/labels_m3.jsonl, entropy/features.jsonl and "
                                f"probes/oof_scores.jsonl; missing: {miss}")
    feats = {r["qid"]: r for r in read_jsonl(paths["entropy"])}
    probes = {r["qid"]: r for r in read_jsonl(paths["probes"])}
    rows, fold_mismatch = [], 0
    for m in read_jsonl(paths["m3"]):
        if m.get("label_m3") not in ("FA", "AH", "UH") or m["qid"] not in feats or m["qid"] not in probes:
            continue
        f, p = feats[m["qid"]], probes[m["qid"]]
        fold_mismatch += int(f["fold"] != m["fold"] or p["fold"] != m["fold"])
        rows.append({"qid": m["qid"], "cls": m["label_m3"], "fold": m["fold"], "relation": m["relation"],
                     "group": m.get("relation_group") or "", "subject": str(m.get("subject_id") or m.get("subject")),
                     "ambiguous_flag": bool((m.get("flags") or {}).get("ambiguous_subject")),
                     **{k: f.get(k) for k in ("tok_mean", "tok_max", "sem_entropy", *LENS, *ATTN)},
                     **{k: v for k, v in p.items() if k.startswith("probe_")}})
    return rows, {"paths": {k: str(v) for k, v in paths.items()}, "fold_mismatch": fold_mismatch}


def matrix(rows: list[dict[str, Any]], cols: list[str], eps: float) -> np.ndarray:
    X = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = (r["probe_sep_sem_entropy"] / (r["tok_mean"] + eps)
                 if c == "ratio_probe_sem_tok" and r.get("probe_sep_sem_entropy") is not None else r.get(c))
            X[i, j] = np.nan if v is None else v
    return X


# ----------------------------------------------------------------------------- models

def _prep(X_tr: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    filled = [np.where(np.isnan(X), med, X) for X in (X_tr, *others)]
    mu, sd = filled[0].mean(0), filled[0].std(0)
    sd[sd < 1e-9] = 1.0
    return [(X - mu) / sd for X in filled]


def fit_predict(kind: str, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, groups_tr: np.ndarray,
                cc: dict[str, Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """P(class) for X_te ([n] for binary = P(y=1), [n, k] for multiclass)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    binary = len(np.unique(y_tr)) == 2
    if kind == "gb":
        m = HistGradientBoostingClassifier(**cc["gb"], class_weight="balanced", random_state=seed)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)
        return (p[:, 1] if binary else p), {}
    # logistic regression: C on an inner subject split (AUROC; macro one-vs-rest for multiclass)
    idx = np.arange(len(y_tr))
    itr, iva = inner_split(idx, groups_tr, float(cc["inner_frac"]), seed)
    best = (-1.0, cc["C_grid"][0])
    for C in cc["C_grid"]:
        A, B = _prep(X_tr[itr], X_tr[iva])
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(A, y_tr[itr])
        p = m.predict_proba(B)
        a = (auroc(p[:, 1], y_tr[iva] == 1) if binary else
             np.nanmean([auroc(p[:, k], y_tr[iva] == c) or np.nan for k, c in enumerate(m.classes_)]))
        if a is not None and a > best[0]:
            best = (float(a), C)
    A, B = _prep(X_tr, X_te)
    m = LogisticRegression(C=best[1], class_weight="balanced", max_iter=2000).fit(A, y_tr)
    p = m.predict_proba(B)
    return (p[:, 1] if binary else p), {"C": best[1], "coef": m.coef_[0].tolist() if binary else None}


def cross_val(kind: str, X: np.ndarray, y: np.ndarray, folds: np.ndarray, groups: np.ndarray,
              cc: dict[str, Any], seed: int, n_classes: int = 2) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = np.full(len(y), np.nan) if n_classes == 2 else np.full((len(y), n_classes), np.nan)
    info = []
    for f in sorted(np.unique(folds)):
        tr, te = folds != f, folds == f
        p, i = fit_predict(kind, X[tr], y[tr], X[te], groups[tr], cc, seed + int(f))
        out[te] = p
        info.append(i)
    return out, info


# ----------------------------------------------------------------------------- metrics

def auprc(score: np.ndarray, pos: np.ndarray) -> float | None:
    """Average precision (step-wise area under the precision-recall curve)."""
    m = ~np.isnan(score)
    s, y = score[m], pos[m].astype(bool)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(prec[y].sum() / y.sum())


def ece(prob: np.ndarray, pos: np.ndarray, bins: int = 10) -> float | None:
    m = ~np.isnan(prob)
    p, y = prob[m], pos[m].astype(float)
    if not len(p):
        return None
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def evaluate(score: np.ndarray, pos: np.ndarray, rows: list[dict[str, Any]], cc: dict[str, Any], seed: int,
             ci: bool = True, prob: bool = False) -> dict[str, Any]:
    rel = np.array([r["relation"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    subj = np.array([r["subject"] for r in rows])
    out: dict[str, Any] = {"pooled": auroc(score, pos), "auprc": auprc(score, pos), "base_rate": float(pos.mean()),
                           "ci": bootstrap_auroc_ci(score, pos, subj, int(cc["n_boot"]), seed) if ci else (None, None),
                           "ece": ece(score, pos) if prob else None}
    within = {}
    for r in sorted(set(rel)):
        k = rel == r
        if (pos & k).sum() >= cc["min_per_class"] and (~pos & k).sum() >= cc["min_per_class"]:
            within[str(r)] = auroc(score[k], pos[k])
    out["within_by_rel"] = within
    out["within"] = float(np.mean(list(within.values()))) if within else None
    for g in ("open", "closed"):
        k = grp == g
        out[g] = auroc(score[k], pos[k]) if (pos & k).any() and (~pos & k).any() else None
    return out


# ----------------------------------------------------------------------------- run

def run(cfg: dict[str, Any], report: Any = None) -> None:
    cc = _cfg(cfg)
    seed = int(cfg.get("seed", 0))
    out = run_dir(cfg) / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows, meta_in = load_rows(cfg)
    cls_all = np.array([r["cls"] for r in all_rows])

    results: dict[str, Any] = {}
    oof_store: dict[str, np.ndarray] = {}
    for task, (neg, posc) in TASKS.items():
        idx = np.flatnonzero(np.isin(cls_all, neg + posc))
        rows = [all_rows[i] for i in idx]
        y = np.isin(cls_all[idx], posc).astype(int)
        folds = np.array([r["fold"] for r in rows])
        groups = np.array([r["subject"] for r in rows])
        res: dict[str, Any] = {"n_neg": int((y == 0).sum()), "n_pos": int(y.sum()), "models": {}, "baselines": {}}
        for s in SETS:
            X = matrix(rows, columns(s), cc["ratio_eps"])
            for kind in ("lr", "gb"):
                p, info = cross_val(kind, X, y, folds, groups, cc, seed)
                main = s == cc["main_set"]
                res["models"][f"{kind}:{s}"] = {**evaluate(p, y == 1, rows, cc, seed, ci=True, prob=True),
                                               "C": [i.get("C") for i in info]}
                if kind == "lr" and main:
                    res["models"][f"{kind}:{s}"]["coef_fold0"] = dict(zip(columns(s), np.round(info[0]["coef"], 3).tolist()))
                oof_store[f"{task}__{kind}:{s}"] = np.full(len(all_rows), np.nan)
                oof_store[f"{task}__{kind}:{s}"][idx] = p
        for name, (col, sign) in BASELINES.items():
            v = sign * np.array([np.nan if r.get(col) is None else r[col] for r in rows], dtype=float)
            res["baselines"][name] = evaluate(v, y == 1, rows, cc, seed)
        rel_ids = {r: k for k, r in enumerate(sorted({r["relation"] for r in rows}))}
        Xr = np.eye(len(rel_ids))[[rel_ids[r["relation"]] for r in rows]]
        p, _ = cross_val("lr", Xr, y, folds, groups, cc, seed)
        res["baselines"]["relation-only"] = evaluate(p, y == 1, rows, cc, seed)
        # sensitivity: main LR model on rows without the heuristic ambiguous_subject flag
        pm = oof_store[f"{task}__lr:{cc['main_set']}"][idx]
        keep = np.array([not r["ambiguous_flag"] for r in rows])
        res["sens_no_ambiguous_flag"] = {"n": int(keep.sum()), "pooled": auroc(pm[keep], y[keep] == 1),
                                         "within": evaluate(pm[keep], y[keep] == 1, [r for r, k in zip(rows, keep) if k],
                                                            cc, seed, ci=False)["within"]}
        res["loro"] = loro(rows, y, matrix(rows, columns(cc["main_set"]), cc["ratio_eps"]), cc, seed) \
            if task in ("hall", "fa_ah") else {}
        results[task] = res
        print(f"[classifier] {task}: done ({time.time() - t0:.0f}s)")

    flat = flat_vs_two_stage(all_rows, cls_all, oof_store, cc, seed)
    timing = inference_time(all_rows, cc, seed, cfg)
    write_jsonl(out / "oof_predictions.jsonl", [
        {"qid": r["qid"], "cls": r["cls"], "fold": r["fold"],
         **{k: (None if np.isnan(v[i]) else float(v[i])) for k, v in oof_store.items() if k.endswith(cc["main_set"])}}
        for i, r in enumerate(all_rows)])
    summary = {"n": len(all_rows), "class_counts": {c: int((cls_all == c).sum()) for c in ("FA", "AH", "UH")},
               "tasks": results, "flat": flat, "timing": timing, "inputs": meta_in,
               "config": {k: v for k, v in cc.items()}, "seconds": time.time() - t0}
    write_json(out / "summary.json", _jsonable(summary))
    write_meta(out, cfg, "classifier", {"summary": {"n": len(all_rows), "class_counts": summary["class_counts"]}})
    if report is not None:
        fill_report(report, summary, cc)
    print(f"[classifier] -> {out} ({time.time() - t0:.0f}s)")


def loro(rows: list[dict[str, Any]], y: np.ndarray, X: np.ndarray, cc: dict[str, Any], seed: int) -> dict[str, Any]:
    """Leave-one-relation-out: train the main LR on all other relations, test on the held-out one."""
    rel = np.array([r["relation"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    out = {}
    for r in sorted(set(rel)):
        te = rel == r
        if (y[te] == 1).sum() < cc["min_per_class"] or (y[te] == 0).sum() < cc["min_per_class"]:
            continue
        p, _ = fit_predict("lr", X[~te], y[~te], X[te], groups[~te], cc, seed)
        out[str(r)] = auroc(p, y[te] == 1)
    return {"by_rel": out, "mean": float(np.mean(list(out.values()))) if out else None}


def flat_vs_two_stage(rows: list[dict[str, Any]], cls: np.ndarray, oof: dict[str, np.ndarray],
                      cc: dict[str, Any], seed: int) -> dict[str, Any]:
    """Flat multinomial LR vs two-stage composition P(FA)=1-p1, P(AH)=p1(1-p2), P(UH)=p1 p2,
    where stage 2 is applied to every row (its out-of-fold model never saw the test fold)."""
    s = cc["main_set"]
    X = matrix(rows, columns(s), cc["ratio_eps"])
    lab = np.array([{"FA": 0, "AH": 1, "UH": 2}[c] for c in cls])
    folds = np.array([r["fold"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    p_flat, _ = cross_val("lr", X, lab, folds, groups, cc, seed, n_classes=3)
    # stage 2 out-of-fold scores exist only for wrong rows; refit per fold to score every row
    p2 = np.full(len(rows), np.nan)
    wrong = np.isin(cls, ["AH", "UH"])
    for f in sorted(np.unique(folds)):
        tr, te = (folds != f) & wrong, folds == f
        p2[te], _ = fit_predict("lr", X[tr], (cls[tr] == "UH").astype(int), X[te], groups[tr], cc, seed + int(f))
    p1 = oof[f"hall__lr:{s}"]
    p_two = np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])
    out = {}
    for name, P in (("flat", p_flat), ("two_stage", p_two)):
        per = {c: evaluate(P[:, k], lab == k, rows, cc, seed, ci=False) for k, c in enumerate(("FA", "AH", "UH"))}
        out[name] = {c: {"pooled": v["pooled"], "within": v["within"]} for c, v in per.items()}
        out[name]["macro_pooled"] = float(np.mean([v["pooled"] for v in per.values()]))
        out[name]["accuracy"] = float((P.argmax(1) == lab).mean())
    return out


def inference_time(rows: list[dict[str, Any]], cc: dict[str, Any], seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Per-question cost of the detector after the greedy pass: 4 linear probes on 2048-d states + the LR,
    vs what the sampled features add (NLI clustering, measured in M2; sampling itself is in the generation time)."""
    rng = np.random.default_rng(seed)
    n, d = 2000, 2048
    H = rng.standard_normal((n, d)).astype(np.float32)
    W = (rng.standard_normal((d, 4)) * 0.01).astype(np.float32)
    t = time.perf_counter()
    for i in range(n):                                   # one question at a time, as at inference
        z = H[i] @ W
        _ = 1 / (1 + np.exp(-z))
    probe_ms = (time.perf_counter() - t) / n * 1e3
    X = matrix(rows[:n], columns(cc["main_set"]), cc["ratio_eps"])
    A = _prep(X)[0]
    w = rng.standard_normal(A.shape[1])
    t = time.perf_counter()
    for i in range(len(A)):
        _ = 1 / (1 + np.exp(-(A[i] @ w)))
    clf_ms = (time.perf_counter() - t) / len(A) * 1e3
    ent = find_upstream(cfg, "entropy/summary.json")
    nli_s = None
    if ent is not None:
        import json
        s = json.loads(ent.read_text(encoding="utf-8"))
        if s.get("nli", {}).get("seconds") and s.get("n"):
            nli_s = s["nli"]["seconds"] / s["n"]
    return {"probes_ms_per_q": probe_ms, "classifier_ms_per_q": clf_ms, "nli_s_per_q": nli_s,
            "note": "CPU numpy timing of the detector head; generation (greedy + 10 samples) 0.74 s/q on a T4 (E03) "
                    "is not split into greedy vs samples — a dedicated timing run is needed for the final H1b table"}


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


# ----------------------------------------------------------------------------- report

def _f(x: float | None) -> str:
    return "  –  " if x is None else f"{x:.3f}"


def _ci(e: dict[str, Any]) -> str:
    lo, hi = e["ci"]
    return f"{_f(e['pooled'])} [{_f(lo)}–{_f(hi)}]" if lo is not None else _f(e["pooled"])


TASK_TITLE = {"hall": "FA vs hallucination (AH+UH)", "ah_uh": "AH vs UH (positive UH)", "fa_ah": "FA vs AH (H2/H3)"}


def fill_report(part: Any, s: dict[str, Any], cc: dict[str, Any]) -> None:
    part.n, part.summary = s["n"], s
    T = s["tasks"]
    main = f"lr:{cc['main_set']}"
    part.hard(s["inputs"]["fold_mismatch"] == 0, f"folds agree across labels, entropy and probe scores "
              f"({s['inputs']['fold_mismatch']} mismatches)")
    for t, r in T.items():
        part.soft(min(r["n_neg"], r["n_pos"]) >= cc["min_per_class"], f"{t}: {r['n_neg']} vs {r['n_pos']} rows")
        a = r["models"][main]["pooled"]
        part.hard(a is not None and a > 0.5, f"{t}: main model AUROC {_f(a)} (> 0.5)")
        e = r["models"][main]
        gap = (e["pooled"] or 0) - (e["within"] or 0)
        part.soft(gap <= cc["max_within_gap"], f"{t}: pooled − within-relation AUROC = {gap:.3f}"
                  + ("" if gap <= cc["max_within_gap"] else " — pooled inflated by relation type (H5)"))
        rel = r["baselines"]["relation-only"]["pooled"] or 0
        part.soft(a - rel >= 0.05, f"{t}: main model beats relation-only by {a - rel:+.3f} pooled")
    best_corr = T["hall"]["baselines"]["correctness probe"]["within"] or 0
    mine = T["hall"]["models"][main]["within"] or 0
    part.soft(mine >= best_corr, f"hall: main model within-relation {mine:.3f} vs direct correctness probe "
              f"{best_corr:.3f}" + ("" if mine >= best_corr else " — the hierarchy does NOT beat the correctness probe"))

    c = s["class_counts"]
    part.key.append(f"rows FA {c['FA']} | AH {c['AH']} | UH {c['UH']} | main feature set {cc['main_set']} | "
                    f"{s['seconds']:.0f}s")
    for t, r in T.items():
        part.key.append(f"{TASK_TITLE[t]} — n {r['n_neg']} / {r['n_pos']}: pooled AUROC [95% CI] | within-relation "
                        "macro | open | closed | AUPRC (base rate) | ECE")
        rows_ = [["model / baseline", "pooled [CI]", "within", "open", "closed", "AUPRC", "ECE"]]
        for k, e in r["models"].items():
            rows_.append([k, _ci(e), _f(e["within"]), _f(e["open"]), _f(e["closed"]),
                          f"{_f(e['auprc'])} ({e['base_rate']:.2f})", _f(e["ece"])])
        for k, e in r["baselines"].items():
            rows_.append([f"baseline: {k}", _ci(e), _f(e["within"]), _f(e["open"]), _f(e["closed"]),
                          f"{_f(e['auprc'])}", ""])
        part.key += table(rows_)
        sn = r["sens_no_ambiguous_flag"]
        part.key.append(f"  without ambiguous_subject-flag rows (n {sn['n']}): pooled {_f(sn['pooled'])}, within "
                        f"{_f(sn['within'])}" + (f" | leave-one-relation-out mean {_f(r['loro']['mean'])}: " + " ".join(
                            f"{k} {_f(v)}" for k, v in r["loro"]["by_rel"].items()) if r["loro"] else ""))
    fl = s["flat"]
    part.key.append("3-class (one-vs-rest AUROC pooled / within): flat multinomial LR vs two-stage")
    part.key += table([["model", "FA", "AH", "UH", "macro pooled", "accuracy"]] + [
        [k, *[f"{_f(v[c]['pooled'])}/{_f(v[c]['within'])}" for c in ("FA", "AH", "UH")], _f(v["macro_pooled"]),
         f"{v['accuracy']:.3f}"] for k, v in fl.items()])
    tm = s["timing"]
    part.key.append(f"inference cost after the greedy pass: probes {tm['probes_ms_per_q']:.3f} ms + classifier "
                    f"{tm['classifier_ms_per_q']:.3f} ms per question (CPU) vs sampled SE: NLI alone "
                    f"{_f(tm['nli_s_per_q'])} s per question (T4) + 10 samples")

    for t, r in T.items():
        part.claude.append(f"{t}: n={r['n_neg']}/{r['n_pos']} " + " | ".join(
            f"{k} {_f(e['pooled'])} w={_f(e['within'])}" for k, e in r["models"].items()))
        part.claude.append(f"  baselines: " + " | ".join(f"{k} {_f(e['pooled'])} w={_f(e['within'])}"
                                                         for k, e in r["baselines"].items()))
        part.claude.append(f"  main within_by_rel={ {k: round(v, 3) for k, v in r['models'][main]['within_by_rel'].items()} } "
                           f"C={r['models'][main]['C']} loro={r['loro']} sens={r['sens_no_ambiguous_flag']}")
        if "coef_fold0" in r["models"][main]:
            part.claude.append(f"  coef fold0: {r['models'][main]['coef_fold0']}")
    part.claude.append(f"flat={fl}")
    part.claude.append(f"timing={ {k: v for k, v in tm.items() if k != 'note'} }")
