"""Stage `entropy` (M2): one feature row per question (CLAUDE.md §3 Entropy hierarchy).

Features
  token      tok_mean, tok_max            per-token entropy of the greedy answer (content tokens)
  sequence   seq_nll                      mean length-normalized NLL of the N samples
  semantic   sem_entropy, sem_n_clusters  bidirectional-entailment clusters of the N samples
                                          (DeBERTa-MNLI, "question answer" strings); abstaining
                                          samples form their own cluster
             abstain_frac                 share of abstaining samples (training-time only)
  layer-wise lens_H_<i>, lens_lstar       logit-lens entropy at the last prompt token, hidden-state
                                          indices from the config; l* = first index where the final
                                          answer's first token is top-1
  attention  attn_mean_<l>, attn_max_<l>  last prompt token → subject tokens, mean / max over heads
  ratio      ratio_sem_tok                sem_entropy / (tok_mean + eps)

Only the semantic part needs the GPU; everything else is read from the saved generation outputs.
Labels must be the M1.6 labels with folds: run `--stages correctness,entropy` so they are
regenerated in this run (the attached dataset holds older labels without folds).

Outputs (outputs/<run>/entropy/): features.jsonl, features.csv, clusters.jsonl, summary.json,
meta.json, shards/ (NLI checkpoints).
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.metrics import auroc, spearman_matrix, vif_from_corr
from ..reporting.console import pct, table
from ..utils import find_upstream, read_jsonl, run_dir, write_json, write_jsonl, write_meta
from .features import (
    attention_features, cluster_by_entailment, lens_features, semantic_entropy, sequence_nll,
    token_features, unique_answers,
)

DEFAULTS = {
    "subset": 200,
    "nli_model": "microsoft/deberta-large-mnli",
    "nli_batch_size": 32,
    "nli_with_question": True,
    "lens_layers": [18, 24, 30, 36],
    "attn_layers": [18, 26, 32],
    "checkpoint_every": 100,
    "ratio_eps": 1e-3,
    "min_per_class": 30,
    "lens_check_max_median_diff": 0.01,
    "lens_check_min_top1": 0.99,
}
ENTROPY_LEVELS = ["tok_mean", "seq_nll", "sem_entropy"]  # + lens_H_<i> added at run time


# ----------------------------------------------------------------------------- inputs

def load_labels(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], Path, bool]:
    """M1.6 labels with folds, plus the path and whether they come from this run's own folder."""
    path = find_upstream(cfg, "correctness/labels.jsonl")
    if path is None:
        raise FileNotFoundError("no correctness/labels.jsonl: run `--stages correctness,entropy`")
    labels = read_jsonl(path)
    if not labels or any(r.get("fold") is None or "label" not in r for r in labels):
        raise RuntimeError(
            f"{path} has no `fold` column — these are pre-M1.6 labels. Run `--stages correctness,entropy` "
            "so the current labels are produced in this run."
        )
    own = str(path).startswith(str(run_dir(cfg)))
    return {r["qid"]: r for r in labels}, path, own


def select_subset(n: int, subset: int | None, seed: int) -> list[int]:
    if not subset or subset >= n:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), subset))


def lens_check(lens_H: np.ndarray, lens_top1: np.ndarray, gens: list[dict[str, Any]]) -> dict[str, float]:
    """Is the last hidden-state index already final-normed? Its lens must reproduce the real
    first-token entropy and argmax of the greedy pass (all questions)."""
    real = np.array([g["greedy"]["token_entropy"][0] for g in gens])
    diff = np.abs(lens_H[:, -1] - real)
    return {"median_abs_diff": float(np.median(diff)), "max_abs_diff": float(diff.max()),
            "top1_rate": float(lens_top1[:, -1].mean())}


# ----------------------------------------------------------------------------- semantic part

def semantic_chunk(gens: list[dict[str, Any]], labs: list[dict[str, Any]], nli: Any, with_q: bool) -> list[dict[str, Any]]:
    """Cluster the samples of a chunk of questions; one batched NLI call for the whole chunk."""
    from .nli import directed_pairs, entail_lookup

    prepared, all_pairs = [], []
    for g, lab in zip(gens, labs):
        texts = [s["text"] for s in g["samples"]]
        abstain = [sl == "abstain" for sl in lab["sample_labels"]]
        reps, idx = unique_answers(texts, abstain)
        pairs = directed_pairs(reps, g["question"] if with_q else None)
        prepared.append((g, reps, idx, len(all_pairs), len(pairs)))
        all_pairs += pairs
    results = nli.entails([(p, h) for _, _, p, h in all_pairs]) if all_pairs else []

    out = []
    for g, reps, idx, start, k in prepared:
        lookup = entail_lookup(all_pairs[start:start + k], results[start:start + k])
        answer_cluster = cluster_by_entailment(len(reps), lookup)
        h, n_clusters, per_sample = semantic_entropy(idx, answer_cluster)
        out.append({"qid": g["qid"], "answers": reps, "answer_cluster": answer_cluster,
                    "sample_cluster": per_sample, "sem_entropy": h, "sem_n_clusters": n_clusters,
                    "nli_pairs": k})
    return out


# ----------------------------------------------------------------------------- stage

def run(cfg: dict[str, Any], report: Any = None) -> None:
    from ..generation.checkpoint import ShardStore, fingerprint

    ecfg = {**DEFAULTS, **(cfg.get("entropy") or {})}
    seed = cfg.get("seed", 0)
    out = run_dir(cfg) / "entropy"

    labels, labels_path, labels_own = load_labels(cfg)
    gen_path = find_upstream(cfg, "generation/generations.jsonl")
    if gen_path is None:
        raise FileNotFoundError("generation/generations.jsonl not found (attach the generation outputs)")
    gens_all = read_jsonl(gen_path)
    with np.load(gen_path.parent / "features.npz") as z:
        lens_H, lens_top1 = z["lens_entropy_tbg"], z["lens_target_top1_tbg"]
        attn_heads = z["attn_subj_heads_tbg"]
    if len(gens_all) != len(lens_H) or any(g["qid"] not in labels for g in gens_all):
        raise RuntimeError("generations, features.npz and labels do not cover the same questions")
    lcheck = lens_check(lens_H, lens_top1, gens_all)

    rows_idx = select_subset(len(gens_all), ecfg["subset"], seed)
    gens = [gens_all[i] for i in rows_idx]
    labs = [labels[g["qid"]] for g in gens]

    # semantic entropy, checkpointed
    fp = fingerprint({"stage": "entropy", "nli": ecfg["nli_model"], "with_q": ecfg["nli_with_question"],
                      "qids": [g["qid"] for g in gens]})
    store = ShardStore(out / "shards", int(ecfg["checkpoint_every"]), fp)
    todo = store.missing_chunks(len(gens))
    t0 = time.perf_counter()
    nli_info: dict[str, Any] = {"model": ecfg["nli_model"], "pairs_this_run": 0}
    if todo:
        from .nli import NLI

        nli = NLI(ecfg["nli_model"], batch_size=int(ecfg["nli_batch_size"]))
        nli_info.update({"device": nli.device, "id2label": nli.id2label, "sanity": nli.sanity()})
        for c in todo:
            r = list(store.chunk_range(c, len(gens)))
            recs = semantic_chunk([gens[i] for i in r], [labs[i] for i in r], nli, bool(ecfg["nli_with_question"]))
            store.save_chunk(c, recs, {"sem_entropy": np.array([x["sem_entropy"] for x in recs])})
            print(f"[entropy] NLI chunk {c + 1}/{store.n_chunks(len(gens))} done")
        nli_info["pairs_this_run"] = nli.n_pairs
        nli_info["switched_to_fp32"] = getattr(nli, "switched_to_fp32", False)
    nli_seconds = time.perf_counter() - t0
    clusters: list[dict[str, Any]] = []
    for c in range(store.n_chunks(len(gens))):
        clusters += store.load_chunk(c)[0]
    if [x["qid"] for x in clusters] != [g["qid"] for g in gens]:
        raise RuntimeError("NLI checkpoints do not match the selected questions")

    # assemble one feature row per question
    feats: list[dict[str, Any]] = []
    for i, g, lab, cl in zip(rows_idx, gens, labs, clusters):
        f: dict[str, Any] = {
            "qid": g["qid"], "label": lab["label"], "relation": lab["relation"],
            "relation_group": lab["relation_group"], "fold": lab["fold"],
            "exclude_from_training": lab["exclude_from_training"], "dataset": lab.get("dataset"),
            **token_features(g["greedy"]),
            "seq_nll": sequence_nll(g["samples"]),
            "sem_entropy": cl["sem_entropy"], "sem_n_clusters": float(cl["sem_n_clusters"]),
            "abstain_frac": lab["abstain_frac"],
            **lens_features(lens_H[i], lens_top1[i], list(ecfg["lens_layers"])),
            **attention_features(attn_heads[i], list(ecfg["attn_layers"])),
        }
        f["ratio_sem_tok"] = f["sem_entropy"] / (f["tok_mean"] + float(ecfg["ratio_eps"]))
        feats.append(f)
    write_jsonl(out / "features.jsonl", feats)
    write_jsonl(out / "clusters.jsonl", clusters)
    with open(out / "features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feats[0]))
        w.writeheader()
        w.writerows(feats)

    summary = summarize(feats, clusters, ecfg)
    summary.update({
        "lens_check": lcheck, "labels_path": str(labels_path), "labels_from_this_run": labels_own,
        "subset": ecfg["subset"], "n_all": len(gens_all), "nli": {**nli_info, "seconds": nli_seconds,
        "chunks": store.n_chunks(len(gens)), "chunks_done_this_run": len(todo)},
    })
    write_json(out / "summary.json", summary)
    write_meta(out, cfg, "entropy", {"summary": {k: summary[k] for k in ("n", "label_counts", "lens_check")}})
    if report is not None:
        fill_report(report, summary, feats, ecfg)
    print(f"[entropy] {len(feats)} questions -> {out}")


# ----------------------------------------------------------------------------- summary + report

FEATURE_COLUMNS_SKIP = {"qid", "label", "relation", "relation_group", "fold", "exclude_from_training", "dataset"}


def feature_names(feats: list[dict[str, Any]]) -> list[str]:
    return [k for k in feats[0] if k not in FEATURE_COLUMNS_SKIP]


def summarize(feats: list[dict[str, Any]], clusters: list[dict[str, Any]], ecfg: dict[str, Any]) -> dict[str, Any]:
    names = feature_names(feats)
    elig = [f for f in feats if not f["exclude_from_training"] and f["label"] in ("correct", "wrong")]
    wrong = np.array([f["label"] == "wrong" for f in elig])
    min_pc = int(ecfg["min_per_class"])

    per_feature = {}
    for k in names:
        col = np.array([f[k] for f in feats], dtype=np.float64)
        e = np.array([f[k] for f in elig], dtype=np.float64)
        within = []
        for rel in sorted({f["relation"] for f in elig}):
            m = np.array([f["relation"] == rel for f in elig])
            if (wrong & m).sum() >= min_pc and (~wrong & m).sum() >= min_pc:
                a = auroc(e[m], wrong[m])
                if a is not None:
                    within.append(a)

        def ms(mask: np.ndarray) -> tuple[float | None, float | None]:
            v = e[mask]
            v = v[~np.isnan(v)]
            return (float(v.mean()), float(v.std())) if v.size else (None, None)

        per_feature[k] = {
            "n_nan": int(np.isnan(col).sum()), "n_inf": int(np.isinf(col).sum()),
            "variance": float(np.nanvar(col)) if (~np.isnan(col)).any() else 0.0,
            "correct": ms(~wrong), "wrong": ms(wrong),
            "auroc_wrong": auroc(e, wrong),
            "auroc_within_macro": float(np.mean(within)) if within else None, "n_relations_within": len(within),
        }

    levels = [k for k in ENTROPY_LEVELS + [f"lens_H_{i}" for i in ecfg["lens_layers"]] if k in names]
    cnames, corr = spearman_matrix({k: np.array([f[k] for f in feats], dtype=np.float64) for k in levels})
    vif = vif_from_corr(corr)

    nc = [c["sem_n_clusters"] for c in clusters]
    labels = [f["label"] for f in feats]
    return {
        "n": len(feats), "n_eligible": len(elig),
        "label_counts": {lab: labels.count(lab) for lab in ("correct", "wrong", "abstain", "partial")},
        "features": per_feature,
        "levels": cnames, "spearman": corr.round(3).tolist(), "vif": [round(float(v), 2) for v in vif],
        "clusters": {"mean": float(np.mean(nc)), "single_cluster_frac": float(np.mean([n == 1 for n in nc])),
                     "nli_pairs": int(sum(c["nli_pairs"] for c in clusters))},
        "subject_missing": int(sum(np.isnan(f[f"attn_mean_{ecfg['attn_layers'][0]}"]) for f in feats)),
    }


def fill_report(part: Any, s: dict[str, Any], feats: list[dict[str, Any]], ecfg: dict[str, Any]) -> None:
    from ..reporting.plots import histograms_by_label

    part.n = s["n"]
    part.summary = s
    lc = s["lens_check"]
    ok = lc["median_abs_diff"] <= ecfg["lens_check_max_median_diff"] and lc["top1_rate"] >= ecfg["lens_check_min_top1"]
    part.hard(ok, f"logit lens at the last hidden-state index reproduces the real first-token distribution "
                  f"(top-1 {100 * lc['top1_rate']:.2f}%, median |ΔH| {lc['median_abs_diff']:.4f}) → index is final-normed"
              if ok else f"logit lens at the last index does NOT match the real output (top-1 {100 * lc['top1_rate']:.2f}%, "
                         f"median |ΔH| {lc['median_abs_diff']:.4f}): layer-wise features are wrong")
    san = s["nli"].get("sanity")
    if san is not None:
        core = san["same_entails"] and san["different_not"]
        part.hard(core, "NLI sanity: same answer entails, different city does not"
                  if core else f"NLI sanity failed: {san} — semantic clusters cannot be trusted")
        part.soft(san["paraphrase_entails"], "NLI sanity: 'Paris' entails 'The city of Paris'"
                  if san["paraphrase_entails"] else "NLI does not see 'Paris' → 'The city of Paris' as entailment: "
                  "paraphrases may split into separate clusters (semantic entropy biased upward)")
    if s["nli"].get("switched_to_fp32"):
        part.soft(False, "NLI produced non-finite fp16 logits and switched to fp32 (slower; results valid)")
    part.hard(s["labels_from_this_run"], "labels with folds produced in this run" if s["labels_from_this_run"]
              else "labels came from an attached dataset, not from this run (risk of stale labels)")
    feats_s = s["features"]
    bad_inf = [k for k, v in feats_s.items() if v["n_inf"]]
    bad_nan = [k for k, v in feats_s.items() if v["n_nan"] and not (k.startswith("attn_") and v["n_nan"] <= s["subject_missing"])]
    part.hard(not bad_inf and not bad_nan, "no NaN/inf (attention NaN only where the subject was not found: "
              f"{s['subject_missing']})" if not bad_inf and not bad_nan else f"NaN/inf in {bad_nan + bad_inf}")
    flat = [k for k, v in feats_s.items() if not v["variance"] > 0]
    part.hard(not flat, "every feature has non-zero variance" if not flat else f"zero variance: {flat}")
    lc_ = s["label_counts"]
    n_corr = sum(1 for f in feats if f["label"] == "correct" and not f["exclude_from_training"])
    part.soft(n_corr >= 20, f"{n_corr} eligible correct answers in the set" + ("" if n_corr >= 20 else " (< 20: AUROCs are very noisy)"))

    n = s["n"]
    part.key.append(f"questions {n} (of {s['n_all']}) | correct {pct(lc_['correct'], n)} | wrong {pct(lc_['wrong'], n)} | "
                    f"abstain {lc_['abstain']} | partial {lc_['partial']} | eligible {s['n_eligible']}")
    cl = s["clusters"]
    part.key.append(f"semantic clusters: mean {cl['mean']:.2f}, single-cluster {100 * cl['single_cluster_frac']:.0f}% | "
                    f"NLI pairs {cl['nli_pairs']} in {s['nli']['seconds']:.0f}s on {s['nli'].get('device', '-')}")

    def f2(x: float | None) -> str:
        return "-" if x is None else f"{x:.3f}"

    def msd(t: tuple[float | None, float | None]) -> str:
        return "-" if t[0] is None else f"{t[0]:.2f} ± {t[1]:.2f}"

    order = sorted(feats_s, key=lambda k: -abs((feats_s[k]["auroc_wrong"] or 0.5) - 0.5))
    part.key.append("features: correct vs wrong (AUROC = P(feature higher for a wrong answer); 0.5 = no signal)")
    part.key += table([["feature", "correct", "wrong", "AUROC", f"within-rel (k)"]] + [
        [k, msd(feats_s[k]["correct"]), msd(feats_s[k]["wrong"]), f2(feats_s[k]["auroc_wrong"]),
         f"{f2(feats_s[k]['auroc_within_macro'])} ({feats_s[k]['n_relations_within']})"] for k in order
    ], indent="  ")
    part.key.append("entropy levels: Spearman correlation | VIF")
    part.key += table([["", *s["levels"], "VIF"]] + [
        [a, *[f"{v:.2f}" for v in row], f"{s['vif'][i]:.1f}"] for i, (a, row) in enumerate(zip(s["levels"], s["spearman"]))
    ], indent="  ")

    part.claude.append(f"n={n}/{s['n_all']} labels={lc_} eligible={s['n_eligible']} labels_path={s['labels_path']}")
    part.claude.append(f"lens_check={ {k: round(v, 4) for k, v in lc.items()} } clusters={cl} nli={ {k: v for k, v in s['nli'].items() if k != 'id2label'} }")
    part.claude.append("auroc " + " ".join(f"{k}={f2(feats_s[k]['auroc_wrong'])}" for k in order))
    part.claude.append(f"vif {dict(zip(s['levels'], s['vif']))}")

    for k in ("tok_mean", "seq_nll", "sem_entropy", "lens_lstar"):
        if k in feats_s:
            part.plots.append((f"{k}: correct vs wrong", histograms_by_label(feats, k)))
