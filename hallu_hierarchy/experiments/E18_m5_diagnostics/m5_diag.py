import json, collections, numpy as np
from hallu_hierarchy.eval.metrics import auroc
from hallu_hierarchy.utils import read_jsonl
R = "hallu_hierarchy/outputs/qwen2.5-3b_popqa_full/"
qids = [g["qid"] for g in read_jsonl(R + "generation/generations.jsonl")]
lab = {r["qid"]: r for r in read_jsonl(R + "correctness/labels.jsonl")}
m3 = {r["qid"]: r["label_m3"] for r in read_jsonl(R + "labeling/labels_m3.jsonl")}
z = np.load(R + "probes/oof_by_source.npz")
cls = np.array([m3.get(q) or "" for q in qids]); rel = np.array([lab[q]["relation"] for q in qids])
grp = np.array([lab[q]["relation_group"] for q in qids]); pred = np.array([lab[q]["pred_norm"] for q in qids])

def within(score, neg, pos, min_pc=30):
    out = []
    for r in sorted(set(rel)):
        k = (rel == r) & np.isin(cls, [neg, pos]) & ~np.isnan(score)
        y = cls[k] == pos
        if y.sum() >= min_pc and (~y).sum() >= min_pc:
            out.append(auroc(score[k], y))
    return np.mean(out) if out else None, len(out)

def show(name, s):
    print(f"{name:28s}", " | ".join(f"{n}={within(s, a, b)[0]:.3f}" for n, (a, b) in
          {"FA<AH": ("FA", "AH"), "FA<UH": ("FA", "UH"), "AH<UH": ("AH", "UH")}.items()))

print("== 1. correctness / SEP probe by position & layer (within-relation macro AUROC)")
for probe in ("correctness", "sep_sem_entropy"):
    for pos in ("tbg", "slt"):
        for l in (12, 20, 24, 28, 36):
            show(f"{probe} {pos}@{l}", z[f"{probe}__{pos}@{l}"])

print("== 2. default answers: how often the model gives this same answer within the relation (leave-one-out)")
freq = np.zeros(len(qids))
for r in set(rel):
    idx = np.flatnonzero(rel == r)
    c = collections.Counter(pred[idx])
    for i in idx:
        freq[i] = (c[pred[i]] - 1) / max(1, len(idx) - 1)
show("answer frequency (higher=AH?)", freq)
for c in ("FA", "AH", "UH"):
    m = cls == c
    print(f"  {c}: median share of the relation answering the same {np.median(freq[m]):.3f}, "
          f">=5% of relation {np.mean(freq[m] >= 0.05):.2f}")

print("== 3. same predicted answer, different label: does the correctness probe still separate FA vs AH?")
s = z["correctness__slt@24"]; t = z["correctness__tbg@32"]
for name, sc in (("slt@24", s), ("tbg@32", t)):
    num = den = 0.0; ngroups = 0; npairs = 0
    for (r, p), idx in collections.defaultdict(list, {}).items(): pass
    groups = collections.defaultdict(list)
    for i in range(len(qids)):
        if cls[i] in ("FA", "AH"): groups[(rel[i], pred[i])].append(i)
    for key, idx in groups.items():
        idx = np.array(idx); y = cls[idx] == "AH"
        if y.sum() and (~y).sum():
            a = auroc(sc[idx], y); w = y.sum() * (~y).sum()
            num += a * w; den += w; ngroups += 1; npairs += int(w)
    print(f"  {name}: pair-weighted AUROC FA vs AH with the SAME answer string = {num / den:.3f} "
          f"({ngroups} (relation, answer) groups, {npairs} pairs)")
