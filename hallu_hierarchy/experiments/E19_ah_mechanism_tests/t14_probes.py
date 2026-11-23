"""E19 tests 1 + 4 (local, needs generation/hidden_slt.npy):
1. Does the correctness probe transfer between closed-set and open-set relations?
4. A probe trained directly on FA vs AH (baseline / upper bound for H3)."""
import collections
import sys

import numpy as np

from hallu_hierarchy.eval.metrics import auroc
from hallu_hierarchy.probes.core import fit_scores, run_probe
from hallu_hierarchy.utils import read_jsonl

R = "hallu_hierarchy/outputs/qwen2.5-3b_popqa_full/"
H = np.load(sys.argv[1], mmap_mode="r")                 # [N, 10, 2048] fp16, indices 0, 4, ..., 36
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
qids = [g["qid"] for g in read_jsonl(R + "generation/generations.jsonl")]
lab = {r["qid"]: r for r in read_jsonl(R + "correctness/labels.jsonl")}
m3 = {r["qid"]: r["label_m3"] for r in read_jsonl(R + "labeling/labels_m3.jsonl")}
L = [lab[q] for q in qids]
cls = np.array([m3.get(q) or "" for q in qids])
rel = np.array([r["relation"] for r in L]); grp = np.array([r["relation_group"] for r in L])
fold = np.array([r["fold"] for r in L]); subj = np.array([str(r["subject_id"]) for r in L])
pred = np.array([r["pred_norm"] for r in L])
gn = np.array([bool(r["flags"].get("gold_noise")) for r in L])
cw = np.isin([r["label"] for r in L], ["correct", "wrong"]) & ~gn
wrong = np.array([r["label"] == "wrong" for r in L])
PC = {"C_grid": [0.001, 0.01, 0.1], "inner_frac": 0.2, "max_iter": 500}


def within(score, mask, pos, mpc=30):
    a = {}
    for r in sorted(set(rel[mask])):
        k = mask & (rel == r) & ~np.isnan(score)
        y = pos[k]
        if y.sum() >= mpc and (~y).sum() >= mpc:
            a[r] = auroc(score[k], y)
    return (float(np.mean(list(a.values()))) if a else float("nan")), a


def X(layer):
    return np.asarray(H[:, LAYERS.index(layer), :], dtype=np.float32)


# ---------------- test 1: closed <-> open transfer of the correctness probe (SLT layer 24, C = 0.001 as in E16)
print("== TEST 1: correctness probe trained on one relation group, tested on the other (SLT@24)")
Xs = X(24)
for tr_g, te_g in (("closed", "open"), ("open", "closed")):
    tr, te = cw & (grp == tr_g), cw & (grp == te_g)
    s = np.full(len(qids), np.nan)
    s[te] = fit_scores(Xs[tr], wrong[tr].astype(int), [Xs[te]], [0.001], 500, 0)[0][0]
    # in-distribution reference: the E16-style 5-fold probe trained on the test group only
    ref = np.full(len(qids), np.nan)
    for f in range(5):
        a, b = te & (fold != f), te & (fold == f)
        ref[b] = fit_scores(Xs[a], wrong[a].astype(int), [Xs[b]], [0.001], 500, 0)[0][0]
    fa_ah = te & np.isin(cls, ["FA", "AH"])
    print(f"  train {tr_g:6s} -> test {te_g:6s}: correct vs wrong within {within(s, te, wrong)[0]:.3f} "
          f"(trained on {te_g} itself: {within(ref, te, wrong)[0]:.3f}) | FA vs AH within "
          f"{within(s, fa_ah, cls == 'AH')[0]:.3f} (itself: {within(ref, fa_ah, cls == 'AH')[0]:.3f})")

# ---------------- test 4: direct FA vs AH probe, nested CV over SLT layers
print("== TEST 4: probe trained directly on FA vs AH (SLT, layers 12-36, nested subject-grouped CV)")
use = np.isin(cls, ["FA", "AH"])
srcs = {f"slt@{l}": (lambda l=l: X(l)) for l in (12, 20, 24, 28, 32, 36)}
res = run_probe("fa_ah", srcs, (cls == "AH").astype(float), use, fold, subj, None, PC, 0)
w, by = within(res.oof_selected, use, cls == "AH")
print(f"  selected per fold {res.selected} | FA vs AH pooled {auroc(res.oof_selected[use], cls[use] == 'AH'):.3f} "
      f"within {w:.3f} | open {auroc(res.oof_selected[use & (grp == 'open')], cls[use & (grp == 'open')] == 'AH'):.3f} "
      f"closed {auroc(res.oof_selected[use & (grp == 'closed')], cls[use & (grp == 'closed')] == 'AH'):.3f}")
print("  by relation:", {k: round(v, 3) for k, v in by.items()})
for s_ in res.sources:
    print(f"   {s_}: within {within(res.oof[s_], use, cls == 'AH')[0]:.3f}")
num = den = 0.0
g = collections.defaultdict(list)
for i in np.flatnonzero(use):
    g[(rel[i], pred[i])].append(i)
for idx in g.values():
    idx = np.array(idx); y = cls[idx] == "AH"
    if y.any() and (~y).any():
        wgt = y.sum() * (~y).sum(); num += auroc(res.oof_selected[idx], y) * wgt; den += wgt
print(f"  same answer string, FA vs AH: {num / den:.3f}  (correctness probe was 0.61)")
