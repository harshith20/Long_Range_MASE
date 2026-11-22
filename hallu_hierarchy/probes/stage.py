"""Stage `probes` (M4): linear probes on cached hidden states (CLAUDE.md §3 Probes). CPU only.

Probes (all logistic regression, standardized inputs, balanced class weights, nested subject-grouped CV;
see core.py):
  entropy levels   SEPs: hidden state at TBG (last prompt token) / SLT (second-last answer token) ->
                   high vs low `sem_entropy` / `seq_nll` (sampled M2 features, binarized per fold at the
                   SEP best split of the training folds). At inference they replace the sampled entropies.
  correctness      direct correctness probe (baseline): TBG / SLT -> wrong vs correct.
  familiarity      H4: subject-token state (prompt-only pass) -> s_pop above the training median.
Layer / position / C are chosen per outer fold on an inner subject-grouped split of the training folds.

Evaluation of the out-of-fold scores (direction: higher = more likely a hallucination; the familiarity
score is flipped, 1 - P(popular)):
  correct vs wrong   pooled with a subject-bootstrap 95% CI, within-relation macro, open / closed
  FA vs UH, FA vs AH, AH vs UH   with the M3 labels (labeling/labels_m3.jsonl), if attached
next to the sampled features they stand in for (sem_entropy, seq_nll) and the greedy token entropy (tok_max).

Inputs: correctness/labels.jsonl (with folds), generation/generations.jsonl + hidden_tbg.npy +
hidden_slt.npy (+ layers.json), subject_states/hidden_subject.npy + index.jsonl, entropy/features.jsonl,
optional labeling/labels_m3.jsonl.
Outputs (outputs/<run>/probes/): oof_scores.jsonl (one row per question: selected-probe scores, for M5),
oof_by_source.npz, summary.json, meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..eval.metrics import auroc, bootstrap_auroc_ci, spearman_matrix
from ..reporting.console import table
from ..utils import find_upstream, read_jsonl, run_dir, write_json, write_jsonl, write_meta
from .core import ProbeResult, run_probe

DEFAULTS: dict[str, Any] = {
    "positions": ["tbg", "slt"],         # generation-pass positions for entropy / correctness probes
    "skip_layers": [0],                  # hidden-state index 0 = embeddings, not a block
    "entropy_targets": ["sem_entropy", "seq_nll"],
    "binarize": "best_split",            # SEP split; familiarity always uses the training median
    "correctness_probe": True,
    "familiarity_probe": True,
    "C_grid": [0.001, 0.01, 0.1, 1.0],
    "inner_frac": 0.2,                   # share of training subjects in the inner validation split
    "max_iter": 500,
    "min_per_class": 30,                 # within-relation AUROC only for relations with >= this per class
    "n_boot": 1000,
    "max_within_gap": 0.10,              # WARN if pooled - within-relation AUROC exceeds this
    "references": ["sem_entropy", "seq_nll", "tok_max"],
}
CONTRASTS = {  # name -> (negative class, positive class); positive = hallucination / less familiar
    "correct_vs_wrong": ("correct", "wrong"),
    "FA_vs_UH": ("FA", "UH"),
    "FA_vs_AH": ("FA", "AH"),
    "AH_vs_UH": ("AH", "UH"),
}


def _cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULTS, **(cfg.get("probes") or {}), "evaluation": cfg.get("evaluation") or {}}


# ----------------------------------------------------------------------------- inputs

def load_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    lab_path = find_upstream(cfg, "correctness/labels.jsonl")
    gen_path = find_upstream(cfg, "generation/generations.jsonl")
    if lab_path is None or gen_path is None:
        raise FileNotFoundError("probes need correctness/labels.jsonl and generation/generations.jsonl "
                                "(run `--stages correctness,probes` with the generation dataset attached)")
    labels = {r["qid"]: r for r in read_jsonl(lab_path)}
    if any(r.get("fold") is None for r in labels.values()):
        raise RuntimeError(f"{lab_path} has no `fold` column (pre-M1.6 labels): run `--stages correctness,probes`")
    qids = [g["qid"] for g in read_jsonl(gen_path)]
    missing = [q for q in qids if q not in labels]
    if missing:
        raise RuntimeError(f"{len(missing)} generations have no label (e.g. {missing[:3]})")

    # The hidden arrays are found on their own: another attached input may hold generations.jsonl without
    # them (e.g. the M3 dataset). Their row order is checked against the generations.jsonl next to them.
    layers = None
    hidden: dict[str, np.ndarray] = {}
    hidden_paths: dict[str, Path] = {}
    order_ok = True
    for pos in ("tbg", "slt"):
        p = find_upstream(cfg, f"generation/hidden_{pos}.npy")
        if p is None:
            continue
        hidden[pos] = np.load(p)                                      # fp16 in RAM (~0.6 GB each for PopQA)
        hidden_paths[pos] = p
        sib = p.parent / "generations.jsonl"
        if sib.exists() and sib != gen_path:
            order_ok &= [g["qid"] for g in read_jsonl(sib)] == qids
        lj = p.parent / "layers.json"
        if layers is None and lj.exists():
            layers = json.loads(lj.read_text())["hidden_state_indices"]
    subj_path = find_upstream(cfg, "subject_states/hidden_subject.npy")
    subj_qids = None
    if subj_path is not None:
        hidden["subject"] = np.load(subj_path)
        idx = subj_path.parent / "index.jsonl"
        subj_qids = [r["qid"] for r in read_jsonl(idx)] if idx.exists() else None
        sl = subj_path.parent / "layers.json"
        if layers is None and sl.exists():
            layers = json.loads(sl.read_text())["hidden_state_indices"]

    feat_path = find_upstream(cfg, "entropy/features.jsonl")
    feats = {r["qid"]: r for r in read_jsonl(feat_path)} if feat_path else {}
    m3_path = find_upstream(cfg, "labeling/labels_m3.jsonl")
    m3 = {r["qid"]: r.get("label_m3") for r in read_jsonl(m3_path)} if m3_path else {}
    return {"labels": labels, "qids": qids, "hidden": hidden, "layers": layers, "subj_qids": subj_qids,
            "feats": feats, "m3": m3, "hidden_order_ok": order_ok,
            "paths": {"labels": lab_path, "generations": gen_path, "subject": subj_path,
                      **{f"hidden_{k}": v for k, v in hidden_paths.items()},
                      "entropy": feat_path, "labels_m3": m3_path}}


def make_sources(arr: np.ndarray, pos: str, layers: list[int], skip: list[int]) -> dict[str, Callable[[], np.ndarray]]:
    """'tbg@24' -> loader of hidden_tbg[:, j(24), :] as float32."""
    return {f"{pos}@{l}": (lambda j=j: np.asarray(arr[:, j, :], dtype=np.float32))
            for j, l in enumerate(layers) if l not in skip}


# ----------------------------------------------------------------------------- evaluation

def evaluate(score: np.ndarray, cls: np.ndarray, neg: str, pos: str, relation: np.ndarray, group: np.ndarray,
             subject: np.ndarray, pc: dict[str, Any], seed: int, ci: bool) -> dict[str, Any]:
    m = ((cls == neg) | (cls == pos)) & ~np.isnan(score)
    y = cls[m] == pos
    s, rel, grp = score[m], relation[m], group[m]
    out: dict[str, Any] = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "pooled": auroc(s, y)}
    out["ci"] = bootstrap_auroc_ci(s, y, subject[m], int(pc["n_boot"]), seed) if ci else (None, None)
    within = []
    mpc = int(pc["min_per_class"])
    for r in sorted(set(rel)):
        k = rel == r
        if (y & k).sum() >= mpc and (~y & k).sum() >= mpc:
            a = auroc(s[k], y[k])
            if a is not None:
                within.append(a)
    out["within"] = float(np.mean(within)) if within else None
    out["n_rel"] = len(within)
    for g in ("open", "closed"):
        k = grp == g
        out[g] = auroc(s[k], y[k]) if (y & k).any() and (~y & k).any() else None
    return out


def target_auroc(res: ProbeResult, scores: np.ndarray) -> float | None:
    m = (res.y >= 0) & ~np.isnan(scores)
    return auroc(scores[m], res.y[m] == 1) if m.any() else None


# ----------------------------------------------------------------------------- run

def run(cfg: dict[str, Any], report: Any = None) -> None:
    pc = _cfg(cfg)
    seed = int(cfg.get("seed", 0))
    out = run_dir(cfg) / "probes"
    out.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(cfg)
    labels, qids, hidden, feats, m3 = inp["labels"], inp["qids"], inp["hidden"], inp["feats"], inp["m3"]
    n = len(qids)
    L = [labels[q] for q in qids]
    folds = np.array([r["fold"] for r in L])
    subject = np.array([str(r.get("subject_id") or r.get("subject")) for r in L])
    relation = np.array([r["relation"] for r in L])
    group = np.array([r.get("relation_group") or "" for r in L])
    label = np.array([r["label"] for r in L])
    gold_noise = np.array([bool((r.get("flags") or {}).get("gold_noise")) for r in L])
    m3_cls = np.array([m3.get(q) or "" for q in qids])

    layers = inp["layers"] or list(range(next(iter(hidden.values())).shape[1])) if hidden else []
    checks = {
        "hidden_rows_ok": all(a.shape[0] == n for a in hidden.values()) and inp["hidden_order_ok"],
        "subject_order_ok": inp["subj_qids"] is None or inp["subj_qids"] == qids,
        "entropy_found": bool(feats),
        "entropy_folds_ok": all(feats[q]["fold"] == labels[q]["fold"] for q in qids if q in feats),
        "m3_found": bool(m3),
        "positions_found": [p for p in pc["positions"] if p in hidden],
    }
    if not checks["hidden_rows_ok"] or not checks["subject_order_ok"]:
        raise RuntimeError(f"hidden-state arrays do not match generations.jsonl ({n} rows): "
                           f"{ {k: a.shape for k, a in hidden.items()} }, subject order ok={checks['subject_order_ok']}")

    gen_sources: dict[str, Callable[[], np.ndarray]] = {}
    for p in checks["positions_found"]:
        gen_sources.update(make_sources(hidden[p], p, layers, pc["skip_layers"]))

    results: dict[str, ProbeResult] = {}
    targets_meta: dict[str, dict[str, Any]] = {}
    if gen_sources and feats:
        use_e = (label != "partial") & ~gold_noise
        for t in pc["entropy_targets"]:
            v = np.array([feats[q][t] if q in feats else np.nan for q in qids], dtype=np.float64)
            u = use_e & ~np.isnan(v)
            results[f"sep_{t}"] = run_probe(f"sep_{t}", gen_sources, v, u, folds, subject, pc["binarize"], pc, seed)
            targets_meta[f"sep_{t}"] = {"target": f"{t} > SEP split", "input": "/".join(checks["positions_found"])}
    if gen_sources and pc["correctness_probe"]:
        u = np.isin(label, ["correct", "wrong"]) & ~gold_noise
        results["correctness"] = run_probe("correctness", gen_sources, (label == "wrong").astype(np.float64), u,
                                           folds, subject, None, pc, seed)
        targets_meta["correctness"] = {"target": "wrong (baseline)", "input": "/".join(checks["positions_found"])}
    if "subject" in hidden and pc["familiarity_probe"]:
        s_pop = np.array([r.get("s_pop") if r.get("s_pop") is not None else np.nan for r in L], dtype=np.float64)
        u = ~gold_noise & ~np.isnan(s_pop)
        src = make_sources(hidden["subject"], "subj", layers, pc["skip_layers"])
        results["familiarity"] = run_probe("familiarity", src, np.log1p(s_pop), u, folds, subject, "median", pc, seed)
        targets_meta["familiarity"] = {"target": "s_pop > median", "input": "subject token"}

    # hallucination scores (higher = more likely hallucinated)
    scores: dict[str, np.ndarray] = {}
    for name, res in results.items():
        scores[f"probe_{name}"] = 1.0 - res.oof_selected if name == "familiarity" else res.oof_selected
    refs = {f"ref_{k}": np.array([feats[q].get(k, np.nan) if q in feats else np.nan for q in qids], dtype=np.float64)
            for k in pc["references"]} if feats else {}

    cls_cw = np.where(np.isin(label, ["correct", "wrong"]) & ~gold_noise, label, "")
    evals: dict[str, dict[str, Any]] = {}
    for name, s in {**scores, **refs}.items():
        evals[name] = {"correct_vs_wrong": evaluate(s, cls_cw, "correct", "wrong", relation, group, subject, pc, seed, True)}
        if m3:
            for c, (neg, pos) in list(CONTRASTS.items())[1:]:
                evals[name][c] = evaluate(s, m3_cls, neg, pos, relation, group, subject, pc, seed, True)

    per_probe: dict[str, Any] = {}
    for name, res in results.items():
        per_probe[name] = {
            **targets_meta[name],
            "n_used": int((res.y >= 0).sum()), "pos_rate": float((res.y == 1).sum() / max(1, (res.y >= 0).sum())),
            "thresholds": res.thresholds, "selected": res.selected,
            "selected_C": {f: res.best_C[s][f] for f, s in res.selected.items()},
            "target_auroc": target_auroc(res, res.oof_selected),
            "by_source": {s: {"target_auroc": target_auroc(res, res.oof[s]),
                              "inner_mean": _mean([a for a in res.inner_auc[s].values() if a is not None]),
                              "wrong_auroc": evaluate(res.oof[s] if name != "familiarity" else 1 - res.oof[s],
                                                      cls_cw, "correct", "wrong", relation, group, subject, pc, seed,
                                                      False)["pooled"]}
                          for s in res.sources},
            "seconds": res.seconds, "n_fits": res.n_fits,
        }
    fam_extra = {}
    if "familiarity" in results:
        f = results["familiarity"].oof_selected
        s_pop = np.array([r.get("s_pop") if r.get("s_pop") is not None else np.nan for r in L], dtype=np.float64)
        _, rho = spearman_matrix({"score": f, "s_pop": s_pop})
        fam_extra["spearman_s_pop"] = float(rho[0, 1])
        fam_extra["mean_by_class"] = {c: _mean(f[(m3_cls if c in ("FA", "AH", "UH") else label) == c])
                                      for c in ("correct", "wrong", "abstain", "FA", "AH", "UH")}

    # outputs for M5
    rows = []
    for i, q in enumerate(qids):
        r = {"qid": q, "fold": int(folds[i]), "label": label[i], "label_m3": m3_cls[i] or None,
             "relation": relation[i], "relation_group": group[i]}
        for name, res in results.items():
            r[f"probe_{name}"] = None if np.isnan(res.oof_selected[i]) else float(res.oof_selected[i])
        rows.append(r)
    write_jsonl(out / "oof_scores.jsonl", rows)
    np.savez_compressed(out / "oof_by_source.npz", **{f"{name}__{s}": res.oof[s] for name, res in results.items()
                                                      for s in res.sources})
    summary = {"n": n, "layers": layers, "checks": checks, "probes": per_probe, "evals": evals,
               "familiarity": fam_extra, "config": {k: v for k, v in pc.items() if k != "evaluation"},
               "paths": {k: str(v) if v else None for k, v in inp["paths"].items()},
               "seconds": float(sum(r.seconds for r in results.values())),
               "n_fits": int(sum(r.n_fits for r in results.values()))}
    write_json(out / "summary.json", _jsonable(summary))
    write_meta(out, cfg, "probes", {"summary": {"n": n, "probes": list(results)}})
    if report is not None:
        fill_report(report, summary, pc)
    print(f"[probes] {list(results)} -> {out}")


def _mean(v: Any) -> float | None:
    v = np.asarray(v, dtype=np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else None


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


def fill_report(part: Any, s: dict[str, Any], pc: dict[str, Any]) -> None:
    part.n, part.summary = s["n"], s
    ck = s["checks"]
    part.hard(ck["entropy_found"], "entropy/features.jsonl found (SEP targets)" if ck["entropy_found"] else
              "entropy/features.jsonl NOT found: SEP probes skipped — attach the M2 entropy outputs")
    if ck["entropy_found"]:
        part.hard(ck["entropy_folds_ok"], "entropy features use the same folds as the labels" if ck["entropy_folds_ok"]
                  else "entropy features have different folds than the labels")
    part.hard(bool(ck["positions_found"]), f"generation hidden states found: {ck['positions_found'] or 'none'}")
    part.soft(ck["m3_found"], "M3 labels found (FA / AH / UH contrasts)" if ck["m3_found"] else
              "labeling/labels_m3.jsonl not attached: FA/AH/UH contrasts skipped")
    for name, p in s["probes"].items():
        a = p["target_auroc"]
        part.hard(a is not None and a > 0.5, f"{name}: out-of-fold AUROC on its own target {_f(a)} (> 0.5)")
    mpc = int(pc["min_per_class"])
    for name, ev in s["evals"].items():
        if not name.startswith("ref_"):
            e = ev["correct_vs_wrong"]
            gap = (e["pooled"] or 0) - (e["within"] or 0)
            part.soft(gap <= pc["max_within_gap"] or e["within"] is None,
                      f"{name}: pooled − within-relation AUROC (correct vs wrong) = {gap:.3f}"
                      + ("" if gap <= pc["max_within_gap"] else " — pooled is inflated by relation type (H5)"))
    first = next(iter(s["evals"].values()), {})
    for c, e in first.items():
        part.soft(min(e["n_pos"], e["n_neg"]) >= mpc, f"{c}: {e['n_neg']} vs {e['n_pos']} out-of-fold rows"
                  + ("" if min(e["n_pos"], e["n_neg"]) >= mpc else f" (< {mpc} in a class)"))

    part.key.append(f"questions {s['n']} | layers {s['layers']} | {s['n_fits']} fits in {s['seconds'] / 60:.1f} min | "
                    f"C grid {pc['C_grid']} | inner split {pc['inner_frac']} of training subjects")
    part.key.append("probes (selected source per outer fold = best inner-split AUROC on the probe's own target)")
    part.key += table([["probe", "target", "input", "n", "pos rate", "target AUROC", "selected (fold: source)"]] + [
        [name, p["target"], p["input"], p["n_used"], f"{p['pos_rate']:.2f}", _f(p["target_auroc"]),
         " ".join(f"{f}:{src}" for f, src in p["selected"].items())] for name, p in s["probes"].items()])
    part.key.append("out-of-fold AUROC by source (target / correct-vs-wrong); * = best target AUROC")
    for name, p in s["probes"].items():
        best = max(p["by_source"], key=lambda k: p["by_source"][k]["target_auroc"] or 0)
        part.key.append(f"  {name}: " + "  ".join(
            f"{src}{'*' if src == best else ''} {_f(v['target_auroc'])}/{_f(v['wrong_auroc'])}"
            for src, v in p["by_source"].items()))
    part.key.append("hallucination detection, correct vs wrong (pooled [95% CI] | within-relation macro | open | closed); "
                    "probe_* = single pass, ref_* = the sampled feature (sem_entropy, seq_nll) or greedy tok_max")
    part.key += table([["score", "pooled [CI]", "within", "n rel", "open", "closed"]] + [
        [name, _ci(ev["correct_vs_wrong"]), _f(ev["correct_vs_wrong"]["within"]), ev["correct_vs_wrong"]["n_rel"],
         _f(ev["correct_vs_wrong"]["open"]), _f(ev["correct_vs_wrong"]["closed"])] for name, ev in s["evals"].items()])
    if ck["m3_found"]:
        part.key.append("M3 contrasts (positive = second class; pooled [CI] / within-relation). H2 expects FA vs AH ≈ 0.5 "
                        "for entropy; H4 expects familiarity to separate UH, not AH")
        cs = [c for c in CONTRASTS if c != "correct_vs_wrong"]
        part.key += table([["score", *cs]] + [
            [name, *[f"{_ci(ev[c])} / {_f(ev[c]['within'])}" for c in cs]] for name, ev in s["evals"].items()])
    if s["familiarity"]:
        fx = s["familiarity"]
        part.key.append(f"familiarity: Spearman(score, s_pop) {fx['spearman_s_pop']:.3f} | mean P(popular) by class "
                        + " ".join(f"{c} {_f(v)}" for c, v in fx["mean_by_class"].items()))

    part.claude.append(f"n={s['n']} layers={s['layers']} fits={s['n_fits']} min={s['seconds'] / 60:.1f} checks={ck}")
    for name, p in s["probes"].items():
        part.claude.append(f"{name}: target_auroc={_f(p['target_auroc'])} n={p['n_used']} pos={p['pos_rate']:.3f} "
                           f"sel={p['selected']} C={p['selected_C']} thr={ {f: round(t, 4) for f, t in p['thresholds'].items()} }")
        part.claude.append(f"  by_source target/wrong: " + " ".join(
            f"{k}={_f(v['target_auroc'])}/{_f(v['wrong_auroc'])}" for k, v in p["by_source"].items()))
    for name, ev in s["evals"].items():
        part.claude.append(f"{name}: " + " | ".join(
            f"{c} {_f(e['pooled'])} ci={tuple(None if x is None else round(x, 3) for x in e['ci'])} w={_f(e['within'])}"
            f"({e['n_rel']}) o={_f(e['open'])} c={_f(e['closed'])} n={e['n_neg']}/{e['n_pos']}" for c, e in ev.items()))
    if s["familiarity"]:
        part.claude.append(f"familiarity={s['familiarity']}")
