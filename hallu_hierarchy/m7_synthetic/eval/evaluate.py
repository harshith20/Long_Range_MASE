"""M7: evaluate checkpoints on qa_test people (plan §4–5, Axes 1, 3, 4).

Cities are single tokens in our BPE (E20), so the answer is decided at the first answer token: the label,
the logit features and the samples all come from one forward pass per prompt.

Per (person, prompt, checkpoint):
  label FA / AH / UH / other (AH = a city linked to the person in the text; `ah_source` says which link)
  Axis 1: token entropy (full vocab), entropy over cities, semantic entropy from 10 samples at T = 1 (exact match)
  Axis 4: p(true city), p(work city), rank of the true city, top-1 vs top-2 margin
Hidden states (all layers, fp16) for the Axis 3 storage probe and the correctness / SEP probes:
  name-only input (Allen-Zhu Q-probing: <eos> + name, last name token), and the last token of the QA prompt.

Usage:
    python -m hallu_hierarchy.m7_synthetic.eval.evaluate --data <corpus dir> --run <run dir> [--ckpt all|last]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROMPTS = {
    "qa": "Q: Where was {name} born?\nA:",                # train question template 0
    "qa_heldout": "Q: {name} was born in which city?\nA:",  # held-out question template (paraphrase check)
    "completion": "{name} was born in",                    # needs no QA extraction (Allen-Zhu "BIO" accuracy)
}
STATE_PROMPT = "qa"


def label_answer(city: str | None, p: dict) -> tuple[str, str]:
    if city is None:
        return "other", ""
    if city == p["birth_city"]:
        return "FA", ""
    if p["n_work_city"] > 0 and city == p["work_city"]:
        return "AH", "work"
    if city == p["university_city"]:
        return "AH", "university"
    if city == p["employer_city"]:
        return "AH", "employer"
    if city in set(p["linked_cities"]):
        return "AH", "partner"
    return "UH", ""


def batched(seqs: list[list[int]], pad: int, device):
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device), (mask.sum(1) - 1).to(device)


@torch.no_grad()
def eval_checkpoint(model, tok, people, cities, device, batch=128, n_samples=10, save_states=True, seed=0):
    city_ids = [tok.encode(" " + c).ids for c in cities]
    assert all(len(c) == 1 for c in city_ids), "cities must be single tokens"
    city_ids = torch.tensor([c[0] for c in city_ids], device=device)
    id2city = {int(i): c for i, c in zip(city_ids.tolist(), cities)}
    eos = tok.token_to_id("<|endoftext|>")
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, states_name, states_prompt = [], [], []
    for kind, fmt in PROMPTS.items():
        for b in range(0, len(people), batch):
            chunk = people[b:b + batch]
            seqs = [tok.encode(fmt.format(name=p["name"])).ids for p in chunk]
            ids, mask, last = batched(seqs, eos, device)
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=save_states and kind == STATE_PROMPT)
            logits = out.logits[torch.arange(len(chunk), device=device), last].float()
            logp = torch.log_softmax(logits, -1)
            prob = logp.exp()
            ent = -(prob * logp).sum(-1)
            pc = prob[:, city_ids]
            pc_n = pc / pc.sum(-1, keepdim=True)
            ent_city = -(pc_n * torch.log(pc_n.clamp_min(1e-12))).sum(-1)
            top2 = prob.topk(2, -1)
            pred = top2.indices[:, 0]
            samples = torch.multinomial(prob, n_samples, replacement=True, generator=gen)
            gold = torch.tensor([tok.encode(" " + p["birth_city"]).ids[0] for p in chunk], device=device)
            work = torch.tensor([tok.encode(" " + p["work_city"]).ids[0] for p in chunk], device=device)
            p_true = prob.gather(1, gold[:, None])[:, 0]
            rank_true = (prob > p_true[:, None]).sum(-1)
            p_work = prob.gather(1, work[:, None])[:, 0]
            for i, p in enumerate(chunk):
                city = id2city.get(int(pred[i]))
                lab, src = label_answer(city, p)
                vals, counts = np.unique(samples[i].cpu().numpy(), return_counts=True)
                fr = counts / counts.sum()
                rows.append({"person_id": int(p["person_id"]), "prompt": kind, "answer": city or tok.decode([int(pred[i])]),
                             "label": lab, "ah_source": src, "tok_entropy": float(ent[i]),
                             "city_entropy": float(ent_city[i]), "sem_entropy_10": float(-(fr * np.log(fr)).sum()),
                             "p_true": float(p_true[i]), "p_work": float(p_work[i]), "rank_true": int(rank_true[i]),
                             "margin": float(top2.values[i, 0] - top2.values[i, 1]),
                             "p_city_mass": float(pc[i].sum())})
            if save_states and kind == STATE_PROMPT:
                hs = torch.stack(out.hidden_states, 1)  # [B, L+1, T, d]
                states_prompt.append(hs[torch.arange(len(chunk), device=device), :, last].half().cpu().numpy())
                nseq = [[eos] + tok.encode(" " + p["name"]).ids for p in chunk]
                nids, nmask, nlast = batched(nseq, eos, device)
                nhs = torch.stack(model(input_ids=nids, attention_mask=nmask, output_hidden_states=True).hidden_states, 1)
                states_name.append(nhs[torch.arange(len(chunk), device=device), :, nlast].half().cpu().numpy())
    states = None
    if save_states:
        states = {"name_last": np.concatenate(states_name), "prompt_last": np.concatenate(states_prompt)}
    return rows, states


def summarize(df, people_df) -> str:
    import pandas as pd
    d = df.merge(people_df[["person_id", "exp_birth", "ratio", "familiarity", "chance_ah"]], on="person_id")
    lines = []
    for kind in PROMPTS:
        x = d[d.prompt == kind]
        acc = x.pivot_table(index="exp_birth", columns=["familiarity", "ratio"], values="label",
                            aggfunc=lambda s: round(100 * (s == "FA").mean())).astype("Int64")
        lines.append(f"\n[{kind}] FA % by exp_birth (rows) x familiarity/ratio (cols)\n{acc.to_string()}")
    x = d[d.prompt == "qa"]
    mix = x.groupby("exp_birth").label.value_counts(normalize=True).unstack(fill_value=0).mul(100).round(1)
    lines.append(f"\n[qa] label mix % by exp_birth\n{mix.to_string()}")
    wrong = x[x.label.isin(["AH", "UH"])]
    if len(wrong):
        ah = wrong.groupby("ratio").label.apply(lambda s: round(100 * (s == "AH").mean(), 1))
        lines.append(f"\n[qa] AH share of wrong answers by ratio (chance ≈ {100 * d.chance_ah.mean():.1f}%)\n{ah.to_string()}")
        src = wrong[wrong.label == "AH"].ah_source.value_counts()
        lines.append(f"\n[qa] AH source\n{src.to_string()}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="all", help="all | last | comma list of checkpoint folder names")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--no-states", action="store_true")
    ap.add_argument("--states-ckpts", default="all", help="checkpoints that save hidden states: all | last")
    args = ap.parse_args(argv)
    import pandas as pd
    from tokenizers import Tokenizer
    from transformers import GPTNeoXForCausalLM

    data, run = Path(args.data), Path(args.run)
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    pdf = pd.read_parquet(data / "people.parquet")
    test = pdf[pdf.split == "qa_test"].sort_values("person_id")
    people = test.to_dict("records")
    cities = sorted(set(pdf.birth_city) | set(pdf.work_city) | set(pdf.university_city) | set(pdf.employer_city))
    ckpts = sorted(run.glob("ckpt_*"))
    if args.ckpt == "last":
        ckpts = ckpts[-1:]
    elif args.ckpt != "all":
        ckpts = [run / c for c in args.ckpt.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = run / "eval"
    out_root.mkdir(exist_ok=True)
    report = []
    for ci, ck in enumerate(ckpts):
        t0 = time.time()
        model = GPTNeoXForCausalLM.from_pretrained(ck).to(device).eval()
        if device.type == "cuda":
            model = model.half()
        save = not args.no_states and (args.states_ckpts == "all" or ci == len(ckpts) - 1)
        rows, states = eval_checkpoint(model, tok, people, cities, device, args.batch, save_states=save)
        df = pd.DataFrame(rows)
        df["checkpoint"] = ck.name
        d = out_root / ck.name
        d.mkdir(exist_ok=True)
        df.to_parquet(d / "results.parquet")
        if states is not None:
            np.savez_compressed(d / "states.npz", person_id=test.person_id.to_numpy(), **states)
        s = summarize(df, pdf)
        report.append(f"\n## {ck.name} ({time.time() - t0:.0f} s)\n{s}")
        print(report[-1], flush=True)
        del model
    (out_root / "SUMMARY.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()
