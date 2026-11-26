"""M7 local analysis of one finished run (E23): CPU only, reads the downloaded Kaggle outputs.

1. learning curves: QA FA % per exp_birth at every checkpoint
2. recency: is exp-1 / exp-3 correctness explained by *when* in the single pass the birth city was seen?
3. completion prompt diagnosis: rerun the last checkpoint with three prompt formats (leading space / <eos>)
4. entropy features at exp 1–3 (the only exposures with errors): AUROC FA vs wrong, FA vs AH, AH vs UH
5. probes on the saved hidden states (last checkpoint, 5-fold CV over people, fixed regularisation):
   a) correctness probe on the last QA-prompt token (exp 1–3)
   b) Q-probe (Allen-Zhu): birth city from the name-only state → stored-but-not-extracted check (H19)
6. AH share of wrong answers by familiarity and ratio (H11, H12)

Usage (from the repo root):
    python -m hallu_hierarchy.m7_synthetic.analysis.local_analysis --run <dir with data/ and train/> --out <dir> --figs <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]  # dataviz reference palette, fixed order
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"


def auroc(y, s, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    if y.min() == y.max():
        return float("nan"), (float("nan"), float("nan")), len(y)
    a = roc_auc_score(y, s)
    if n_boot == 0:
        return float(a), (float("nan"), float("nan")), len(y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], s[i]))
    return float(a), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), len(y)


def fmt_auc(r):
    a, (lo, hi), n = r
    return f"{a:.3f} [{lo:.3f}, {hi:.3f}] n={n}"


# ---------------------------------------------------------------- 1. learning curves
def learning_curves(run: Path, ppl: pd.DataFrame) -> pd.DataFrame:
    log = [json.loads(l) for l in (run / "train" / "train_log.jsonl").read_text().splitlines() if l.strip()]
    tok = {r["checkpoint"]: r["tokens_seen"] for r in log if "checkpoint" in r}
    rows = []
    for f in sorted((run / "train" / "eval").glob("ckpt_*/results.parquet")):
        r = pd.read_parquet(f).merge(ppl[["person_id", "exp_birth"]], on="person_id")
        ck = f.parent.name
        for (prompt, e), g in r.groupby(["prompt", "exp_birth"]):
            rows.append({"checkpoint": ck, "tokens_M": tok.get(ck, np.nan) / 1e6, "prompt": prompt, "exp_birth": e,
                         "fa": (g.label == "FA").mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. recency
def recency(run: Path, ppl: pd.DataFrame, last: pd.DataFrame, total_tokens: int) -> tuple[pd.DataFrame, str]:
    import pyarrow.parquet as pq
    ids = set(ppl[(ppl.split == "qa_test") & ppl.exp_birth.isin([1, 3])].person_id)
    t = pq.read_table(run / "data" / "exposure_log.parquet",
                      columns=["person_id", "attribute", "token_offset", "is_exposure", "context_type"]).to_pandas()
    t = t[(t.attribute == "birth_city") & t.is_exposure & t.person_id.isin(ids)]
    pos = t.groupby("person_id").token_offset.agg(["max", "min", "count"]).reset_index()
    pos["last_pos"] = pos["max"] / total_tokens
    q = last[last.prompt == "qa"][["person_id", "label"]].merge(pos, on="person_id").merge(
        ppl[["person_id", "exp_birth", "n_birth_city"]], on="person_id")
    q["fa"] = (q.label == "FA").astype(int)
    q["bin"] = pd.cut(q.last_pos, [0, .2, .4, .6, .8, .9, 1.0])
    tab = q.groupby(["exp_birth", "bin"], observed=True).fa.agg(["mean", "size"]).reset_index()
    lines = [f"exposure count check: logged birth exposures = n_birth_city for "
             f"{(q['count'] == q.n_birth_city).mean():.1%} of people"]
    for e in (1, 3):
        g = q[q.exp_birth == e]
        a = auroc(g.fa, g.last_pos)
        lines.append(f"exp {e}: AUROC(position of last exposure → FA) {fmt_auc(a)}; "
                     f"FA last 10% of corpus {g[g.last_pos > .9].fa.mean():.1%} (n={(g.last_pos > .9).sum()}) vs "
                     f"first 50% {g[g.last_pos <= .5].fa.mean():.1%} (n={(g.last_pos <= .5).sum()})")
    return tab, "\n".join(lines)


# ---------------------------------------------------------------- 3. completion diagnosis
@torch.no_grad()
def completion_variants(run: Path, ppl: pd.DataFrame, cities: list[str]) -> tuple[pd.DataFrame, str]:
    from tokenizers import Tokenizer
    from transformers import GPTNeoXForCausalLM
    tok = Tokenizer.from_file(str(run / "data" / "tokenizer.json"))
    ck = sorted((run / "train").glob("ckpt_*"))[-1]
    model = GPTNeoXForCausalLM.from_pretrained(ck).eval()
    eos = tok.token_to_id("<|endoftext|>")
    cid = torch.tensor([tok.encode(" " + c).ids[0] for c in cities])
    test = ppl[ppl.split == "qa_test"].sort_values("person_id")
    gold = torch.tensor([cities.index(c) for c in test.birth_city])
    variants = {
        "as evaluated: '{name} was born in'": lambda n: tok.encode(f"{n} was born in").ids,
        "<eos> + '{name} was born in' (= document start)": lambda n: [eos] + tok.encode(f"{n} was born in").ids,
        "<eos> + ' {name} was born in' (mid-text name tokens)": lambda n: [eos] + tok.encode(f" {n} was born in").ids,
        "QA prompt (control)": lambda n: tok.encode(f"Q: Where was {n} born?\nA:").ids,
    }
    rows, top_other = [], {}
    names = test.name.tolist()
    for vname, fn in variants.items():
        top1, city_top1, mass = [], [], []
        nxt = []
        for b in range(0, len(names), 256):
            seqs = [fn(n) for n in names[b:b + 256]]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), eos)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s)
                mask[i, :len(s)] = 1
            lp = model(input_ids=ids, attention_mask=mask).logits[torch.arange(len(seqs)), mask.sum(1) - 1].float()
            pr = lp.softmax(-1)
            pc = pr[:, cid]
            g = gold[b:b + 256]
            top1.append(pr.argmax(-1) == cid[g])
            city_top1.append(pc.argmax(-1) == g)
            mass.append(pc.sum(-1))
            nxt.append(pr.argmax(-1))
        top1, city_top1, mass, nxt = map(torch.cat, (top1, city_top1, mass, nxt))
        is_city = torch.isin(nxt, cid)
        top_other[vname] = pd.Series([tok.decode([int(i)]) for i in nxt[~is_city]]).value_counts().head(5).to_dict()
        for e in sorted(test.exp_birth.unique()):
            m = torch.tensor((test.exp_birth == e).to_numpy())
            rows.append({"variant": vname, "exp_birth": int(e), "fa_top1": float(top1[m].float().mean()),
                         "fa_city_argmax": float(city_top1[m].float().mean()), "p_city_mass": float(mass[m].mean())})
    lines = [f"most frequent non-city next tokens: {json.dumps(top_other, ensure_ascii=False)}"]
    return pd.DataFrame(rows), "\n".join(lines)


# ---------------------------------------------------------------- 4. entropy AUROC
def entropy_aurocs(last: pd.DataFrame, ppl: pd.DataFrame) -> str:
    q = last[last.prompt == "qa"].merge(ppl[["person_id", "exp_birth"]], on="person_id")
    q = q[q.exp_birth.isin([1, 3]) & q.label.isin(["FA", "AH", "UH"])].copy()
    q["neg_margin"] = -q.margin
    feats = ["tok_entropy", "city_entropy", "sem_entropy_10", "neg_margin"]
    lines = [f"QA prompt, exp 1–3, n = {len(q)}: " + ", ".join(f"{k} {v}" for k, v in q.label.value_counts().items()),
             "AUROC, higher feature = the second class (95% bootstrap CI):"]
    tasks = {"FA vs wrong (AH+UH)": (["FA"], ["AH", "UH"]), "FA vs AH": (["FA"], ["AH"]), "AH vs UH": (["AH"], ["UH"])}
    for tname, (a, b) in tasks.items():
        lines.append(f"  {tname}")
        for f in feats:
            g = q[q.label.isin(a + b)]
            pooled = auroc(g.label.isin(b).astype(int), g[f])
            within = [auroc(h.label.isin(b).astype(int), h[f], n_boot=0)[0] for _, h in g.groupby("exp_birth")]
            lines.append(f"    {f:15s} pooled {fmt_auc(pooled)} | within exp 1 / 3: "
                         + " / ".join(f"{w:.3f}" for w in within))
    return "\n".join(lines)


# ---------------------------------------------------------------- 5. probes
def _folds(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.permutation(n) % k


def softmax_probe(X, y, n_cls, folds, wd=1e-3, iters=200):
    """Multinomial logistic regression, full-batch L-BFGS, standardised inputs. Returns out-of-fold argmax."""
    pred = np.zeros(len(y), dtype=int)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        ytr = torch.tensor(y[tr])
        W = torch.zeros(X.shape[1], n_cls, requires_grad=True)
        b = torch.zeros(n_cls, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + wd * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pred[te] = (Xte @ W + b).argmax(-1).numpy()
    return pred


def probes(run: Path, last: pd.DataFrame, ppl: pd.DataFrame, cities: list[str]) -> tuple[str, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    z = np.load(sorted((run / "train" / "eval").glob("ckpt_*/states.npz"))[-1], allow_pickle=False)
    pid = z["person_id"]
    meta = pd.DataFrame({"person_id": pid}).merge(ppl[["person_id", "exp_birth", "birth_city"]], on="person_id")
    q = last[last.prompt == "qa"].set_index("person_id").loc[pid]
    meta["qa_fa"] = (q.label.to_numpy() == "FA")
    n_layers = z["name_last"].shape[1]
    lines = []

    # a) correctness probe at exp 1–3 (last prompt token)
    sel = meta.exp_birth.isin([1, 3]).to_numpy()
    y = meta.qa_fa.to_numpy()[sel].astype(int)
    folds = _folds(sel.sum())
    lines.append(f"(a) correctness probe, last QA-prompt token, exp 1–3 (n={sel.sum()}, FA {y.mean():.1%}), "
                 "5-fold CV, logistic regression C=0.05:")
    best = None
    for L in range(1, n_layers):
        X = z["prompt_last"][sel, L].astype(np.float32)
        s = np.zeros(len(y))
        for f in range(5):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            clf.fit(X[folds != f], y[folds != f])
            s[folds == f] = clf.predict_proba(X[folds == f])[:, 1]
        a = auroc(1 - y, -s, n_boot=0)[0]  # FA vs wrong, same orientation as the entropy table
        lines.append(f"    layer {L:2d}: AUROC FA vs wrong {a:.3f}")
        if best is None or a > best[1]:
            best = (L, a, s)
    s = best[2]
    a = auroc(1 - y, -s)
    lines.append(f"    best layer {best[0]} (chosen on the test folds → optimistic by a little): {fmt_auc(a)}")

    # b) Q-probe: birth city from the name-only state, all qa_test people
    yc = np.array([cities.index(c) for c in meta.birth_city])
    folds = _folds(len(yc), seed=1)
    lines.append(f"(b) Q-probe: birth city ({len(set(yc))} classes, chance ≈ {1 / len(set(yc)):.1%}) from the "
                 "<eos> + name state (last name token), 5-fold CV over people:")
    rows = []
    for L in range(2, n_layers, 2):
        pred = softmax_probe(z["name_last"][:, L].astype(np.float32), yc, len(cities), folds)
        ok = pred == yc
        for e in sorted(meta.exp_birth.unique()):
            m = (meta.exp_birth == e).to_numpy()
            rows.append({"layer": L, "exp_birth": int(e), "probe_acc": ok[m].mean(), "qa_acc": meta.qa_fa[m].mean(),
                         "probe_ok_given_qa_wrong": ok[m & ~meta.qa_fa.to_numpy()].mean()
                         if (m & ~meta.qa_fa.to_numpy()).any() else np.nan,
                         "n_qa_wrong": int((m & ~meta.qa_fa.to_numpy()).sum())})
        lines.append(f"    layer {L:2d}: overall {ok.mean():.1%}")

    # c) logit lens on the name-only state: does the model's own unembedding read the birth city off the name?
    from transformers import GPTNeoXForCausalLM
    from tokenizers import Tokenizer
    model = GPTNeoXForCausalLM.from_pretrained(sorted((run / "train").glob("ckpt_*"))[-1]).eval()
    tok = Tokenizer.from_file(str(run / "data" / "tokenizer.json"))
    cid = torch.tensor([tok.encode(" " + c).ids[0] for c in cities])
    lines.append(f"(c) logit lens on the name-only state, rank of the birth city among {len(cities)} cities "
                 "(0 = top-1; random ≈ 100):")
    with torch.no_grad():
        for L in range(4, n_layers, 2):
            h = torch.tensor(z["name_last"][:, L].astype(np.float32))
            if L < n_layers - 1:  # the last saved state already has the final layer norm
                h = model.gpt_neox.final_layer_norm(h)
            lc = model.embed_out(h)[:, cid]
            rank = (lc > lc[torch.arange(len(yc)), torch.tensor(yc)][:, None]).sum(1).numpy()
            lines.append(f"    layer {L:2d}: top-1 {np.mean(rank == 0):.1%}, median rank {np.median(rank):.0f}")
    return "\n".join(lines), pd.DataFrame(rows)


# ---------------------------------------------------------------- 6. AH share by familiarity
def ah_tables(last: pd.DataFrame, ppl: pd.DataFrame) -> str:
    q = last[last.prompt == "qa"].merge(ppl[["person_id", "exp_birth", "ratio", "familiarity", "chance_ah"]],
                                        on="person_id")
    w = q[q.label.isin(["AH", "UH"])]
    t = w.assign(ah=(w.label == "AH")).pivot_table(index="familiarity", columns="ratio", values="ah",
                                                    aggfunc=["mean", "size"])
    t2 = w.assign(ah=(w.label == "AH")).pivot_table(index="exp_birth", columns="ratio", values="ah", aggfunc="mean")
    return (f"AH share of wrong answers (AH+UH) by familiarity × ratio (mean, then n):\n{t.round(3).to_string()}\n"
            f"by exp_birth × ratio:\n{t2.round(3).to_string()}\nchance (mean over wrong rows): {w.chance_ah.mean():.3f}")


# ---------------------------------------------------------------- figures
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def figures(figs: Path, curves, comp, ah_by_ratio, rec, chance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs.mkdir(parents=True, exist_ok=True)

    # accuracy vs exposure, three measures
    fig, ax = plt.subplots(figsize=(6.4, 4))
    lastck = curves.checkpoint.max()
    c = curves[curves.checkpoint == lastck]
    exps = sorted(c.exp_birth.unique())
    series = [("QA (train question)", c[c.prompt == "qa"].set_index("exp_birth").fa),
              ("QA (held-out paraphrase)", c[c.prompt == "qa_heldout"].set_index("exp_birth").fa)]
    best = comp[comp.variant.str.startswith("<eos> + ' ")].set_index("exp_birth").fa_city_argmax
    series.append(("Completion (most likely city)", best))
    for i, (lab, s) in enumerate(series):
        ax.plot(range(len(exps)), [100 * s[e] for e in exps], color=SERIES[i], lw=2, marker="o", ms=6, label=lab)
    ax.set_xticks(range(len(exps)), [str(e) for e in exps])
    ax.set_ylim(0, 105)
    _style(ax, "Accuracy vs exposure (last checkpoint)", "times the birth city was seen", "correct (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "accuracy_vs_exposure.png", dpi=200)
    plt.close(fig)

    # AH share by ratio
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    r = ah_by_ratio
    ax.bar(range(len(r)), 100 * r.values, color=SERIES[0], width=0.6)
    for i, v in enumerate(r.values):
        ax.text(i, 100 * v + 0.8, f"{100 * v:.0f}%", ha="center", color=INK, fontsize=10)
    ax.axhline(100 * chance, color=INK2, lw=1.2, ls="--")
    ax.set_xlim(-0.5, len(r) + 0.4)
    ax.text(len(r) - 0.35, 100 * chance, f"chance\n{100 * chance:.1f}%", ha="left", va="center", color=INK2, fontsize=9)
    ax.set_xticks(range(len(r)), [str(x) for x in r.index])
    _style(ax, "Wrong answers that are a linked city (AH)", "competitor ratio (work-city ÷ birth-city mentions)",
           "AH share of wrong answers (%)")
    fig.tight_layout()
    fig.savefig(figs / "ah_vs_ratio.png", dpi=200)
    plt.close(fig)

    # learning curves
    fig, ax = plt.subplots(figsize=(7.2, 4))
    q = curves[curves.prompt == "qa"]
    for i, e in enumerate(exps):
        g = q[q.exp_birth == e].sort_values("tokens_M")
        ax.plot(g.tokens_M, 100 * g.fa, color=SERIES[i], lw=2, marker="o", ms=4, label=f"exp {e}")
    ax.set_ylim(0, 105)
    _style(ax, "QA accuracy during training, by exposure", "training tokens seen (M)", "correct (%)")
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(figs / "learning_curves.png", dpi=200)
    plt.close(fig)

    # recency
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, e in enumerate([1, 3]):
        g = rec[rec.exp_birth == e]
        x = [iv.right for iv in g.bin]
        ax.plot(x, 100 * g["mean"], color=SERIES[i], lw=2, marker="o", ms=6, label=f"exp {e}")
    ax.set_ylim(0, 105)
    _style(ax, "Recency: when the last birth-city sentence was seen", "position in the single pass (bin end)",
           "QA correct (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "recency.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", required=True)
    args = ap.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ppl = pd.read_parquet(run / "data" / "people.parquet")
    budget = json.loads((run / "data" / "budget.json").read_text())
    total = budget["measured_this_run"]["tokens_total"]
    cities = sorted(set(ppl.birth_city) | set(ppl.work_city) | set(ppl.university_city) | set(ppl.employer_city))
    last_file = sorted((run / "train" / "eval").glob("ckpt_*/results.parquet"))[-1]
    last = pd.read_parquet(last_file)
    rep = [f"# M7 local analysis — {run.name}, last checkpoint {last_file.parent.name}\n"]

    curves = learning_curves(run, ppl)
    curves.to_csv(out / "learning_curves.csv", index=False)
    piv = curves[curves.prompt == "qa"].pivot_table(index="tokens_M", columns="exp_birth", values="fa").mul(100).round(1)
    rep.append(f"## 1. Learning curves (QA FA %, rows = tokens seen, M)\n{piv.to_string()}\n")

    rec, rtxt = recency(run, ppl, last, total)
    rec.to_csv(out / "recency.csv", index=False)
    rep.append(f"## 2. Recency (exp 1 and 3, qa_test)\n{rtxt}\n{rec.round(3).to_string()}\n")

    comp, ctxt = completion_variants(run, ppl, cities)
    comp.to_csv(out / "completion_variants.csv", index=False)
    cp = comp.pivot_table(index="variant", columns="exp_birth", values="fa_top1").mul(100).round(1)
    cm = comp.pivot_table(index="variant", columns="exp_birth", values="p_city_mass").round(2)
    ca = comp.pivot_table(index="variant", columns="exp_birth", values="fa_city_argmax").mul(100).round(1)
    rep.append(f"## 3. Completion prompt diagnosis (FA % by exp_birth, top-1 token)\n{cp.to_string()}\n"
               f"p(any city) at the answer position:\n{cm.to_string()}\n{ctxt}\n")

    rep.append(f"## 4. Entropy features at exp 1–3\n{entropy_aurocs(last, ppl)}\n")

    ptxt, qprobe = probes(run, last, ppl, cities)
    qprobe.to_csv(out / "qprobe.csv", index=False)
    best_layer = qprobe.groupby("layer").probe_acc.mean().idxmax()
    qb = qprobe[qprobe.layer == best_layer].set_index("exp_birth").round(3)
    rep.append(f"## 5. Probes\n{ptxt}\nQ-probe at layer {best_layer} by exp_birth:\n{qb.to_string()}\n")

    rep.append(f"## 6. AH share\n{ah_tables(last, ppl)}\n")

    q = last[last.prompt == "qa"].merge(ppl[["person_id", "ratio", "chance_ah"]], on="person_id")
    w = q[q.label.isin(["AH", "UH"])]
    figures(Path(args.figs), curves, comp, w.groupby("ratio").label.apply(lambda s: (s == "AH").mean()), rec,
            w.chance_ah.mean())
    text = "\n".join(rep)
    (out / "ANALYSIS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()
