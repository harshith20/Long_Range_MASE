import numpy as np
from hallu_hierarchy.classifier import stage as S
from hallu_hierarchy.utils import load_config
cfg = load_config("configs/qwen3b_popqa_full.yaml"); cc = S._cfg(cfg)
rows, _ = S.load_rows(cfg)
cls = np.array([r["cls"] for r in rows]); lab = np.array([{"FA": 0, "AH": 1, "UH": 2}[c] for c in cls])
folds = np.array([r["fold"] for r in rows]); groups = np.array([r["subject"] for r in rows])
S.BLOCKS["S"] = ["probe_sep_sem_entropy"]
for s in ("S", "C", "S+C", "E+L+A+F", "E+L+A+F+C"):
    X = S.matrix(rows, S.columns(s), cc["ratio_eps"])
    for kind in ("lr", "gb"):
        P, _ = S.cross_val(kind, X, lab, folds, groups, cc, 0, n_classes=3)
        per = {c: S.evaluate(P[:, k], lab == k, rows, cc, 0, ci=False) for k, c in enumerate(("FA", "AH", "UH"))}
        pred = P.argmax(1)
        rec = {c: float((pred[lab == k] == k).mean()) for k, c in enumerate(("FA", "AH", "UH"))}
        print(f"{kind}:{s:10s} " + " ".join(f"{c} {v['pooled']:.3f}/{v['within']:.3f}" for c, v in per.items())
              + f" | macro within {np.mean([v['within'] for v in per.values()]):.3f} | recall " + " ".join(f"{c} {v:.2f}" for c, v in rec.items()))
# where do AH rows sit in the 2-D (SEP, correctness) plane?
sep = np.array([r["probe_sep_sem_entropy"] for r in rows]); cor = np.array([r["probe_correctness"] for r in rows])
for c in ("FA", "AH", "UH"):
    m = cls == c
    hi_s, hi_c = sep[m] > np.median(sep), cor[m] > np.median(cor)
    print(f"{c}: median SEP {np.median(sep[m]):.2f} correctness {np.median(cor[m]):.2f} | quadrants "
          f"lowSEP+lowWrong {np.mean(~hi_s & ~hi_c):.2f} lowSEP+highWrong {np.mean(~hi_s & hi_c):.2f} "
          f"highSEP+lowWrong {np.mean(hi_s & ~hi_c):.2f} highSEP+highWrong {np.mean(hi_s & hi_c):.2f}")
