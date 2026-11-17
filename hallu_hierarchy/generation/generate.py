"""Stage `generation`: greedy answer + N samples per question, with caching.

Per question we save:
  * greedy answer text/tokens, per-token entropy and log-prob (raw logits, T=1)
  * N sampled answers with per-token log-probs
  * hidden states of the greedy pass at TBG and SLT for a subset of layers (fp16)
  * logit-lens entropy / target log-prob / target-is-top1 at TBG and SLT for every layer
  * attention mass on subject tokens (mean over heads) for every layer
  * per head, at the last prompt token (TBG): attention mass on subject tokens for every layer
    (attn_subj_heads_tbg [L, H]) and the full attention row at a few middle/late layers
    (attn_row_tbg [len(attn_row_layers), H, attn_row_max_len], NaN-padded). These give the
    mean- and max-over-heads attention features of M2.

Positions (indices into prompt + greedy answer):
  TBG = last prompt token (state before the first answer token is produced)
  SLT = second-last token of the generated sequence. When the answer ends with EOS this is
        the last real answer token (as in the SEP paper). If only one token was generated,
        SLT coincides with TBG.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ..data.splits import check_no_leakage
from ..reporting.console import pct
from ..reporting.plots import histogram
from ..utils import read_jsonl, run_dir, set_seed, write_json, write_jsonl, write_meta
from .checkpoint import ShardStore, fingerprint
from .prompts import encode_prompt

HIDDEN_KEYS = ("hidden_tbg", "hidden_slt")


# ----------------------------------------------------------------------------- model

def load_model(mcfg: dict[str, Any]) -> tuple[Any, Any, set[int]]:
    dtype = getattr(torch, mcfg.get("dtype", "float16"))
    device = mcfg.get("device", "cuda")
    tok = AutoTokenizer.from_pretrained(mcfg["id"], revision=mcfg.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["id"],
        revision=mcfg.get("revision"),
        torch_dtype=dtype,
        attn_implementation="eager",  # needed to return attention weights
    ).to(device)
    model.eval()

    # Replace the model's generation defaults (Qwen ships top_p/top_k/repetition_penalty)
    # with a clean config so only the values in our YAML take effect.
    base = model.generation_config
    eos = base.eos_token_id if base.eos_token_id is not None else tok.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    pad = base.pad_token_id if base.pad_token_id is not None else tok.pad_token_id
    if pad is None:
        pad = min(eos_ids)
    model.generation_config = GenerationConfig(
        bos_token_id=base.bos_token_id, eos_token_id=sorted(eos_ids), pad_token_id=pad
    )
    return model, tok, eos_ids


def final_norm_and_head(model: Any) -> tuple[Any, Any]:
    backbone = getattr(model, "model", None)
    norm = getattr(backbone, "norm", None)
    if norm is None:
        raise AttributeError(f"don't know the final norm of {type(model).__name__} for the logit lens")
    return norm, model.get_output_embeddings()


def selected_layers(n_layers: int, stride: int) -> list[int]:
    """Hidden-state indices to cache: 0 (embeddings), every `stride`-th layer, and the last."""
    return sorted(set(range(0, n_layers + 1, stride)) | {n_layers})


# ----------------------------------------------------------------------------- helpers

def truncate_at_eos(tokens: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
    """Keep tokens up to and including the first EOS."""
    for i, t in enumerate(tokens):
        if t in eos_ids:
            return tokens[: i + 1], True
    return tokens, False


def decode_answer(tok: Any, tokens: list[int], ends_with_eos: bool) -> str:
    body = tokens[:-1] if ends_with_eos else tokens
    return tok.decode(body, skip_special_tokens=True).strip()


@torch.no_grad()
def logit_lens(hs: tuple[torch.Tensor, ...], pos: int, target: int, norm: Any, head: Any) -> dict[str, np.ndarray]:
    """Logit lens at one position for every layer. The last hidden state is already normed."""
    h = torch.stack([layer[0, pos] for layer in hs])  # [L+1, d]
    h = torch.cat([norm(h[:-1]), h[-1:]], dim=0)
    logp = F.log_softmax(head(h).float(), dim=-1)
    return {
        "entropy": (-(logp.exp() * logp).sum(-1)).cpu().numpy().astype(np.float32),
        "target_logprob": logp[:, target].cpu().numpy().astype(np.float32),
        "target_top1": (logp.argmax(-1) == target).cpu().numpy(),
    }


@torch.no_grad()
def subject_attention(
    attentions: tuple[torch.Tensor, ...], subj_idx: list[int], queries: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    """Attention mass on subject tokens (sum over subject keys, mean over heads) per layer."""
    n_layers = len(attentions)
    if not subj_idx:
        return {k: np.full(n_layers, np.nan, dtype=np.float32) for k in queries}
    mass = torch.stack([a[0].float().mean(0)[:, subj_idx].sum(-1) for a in attentions])  # [L, S]
    return {k: mass[:, q].mean(-1).cpu().numpy().astype(np.float32) for k, q in queries.items()}


def attention_row_layers(n_layers: int, fracs: list[float]) -> list[int]:
    """Indices into the attentions tuple (0..n_layers-1) at the given depth fractions."""
    return sorted({min(n_layers - 1, max(0, round(f * (n_layers - 1)))) for f in fracs})


@torch.no_grad()
def tbg_attention_per_head(
    attentions: tuple[torch.Tensor, ...], subj_idx: list[int], q_pos: int,
    row_layers: list[int], max_len: int,
) -> tuple[dict[str, np.ndarray], bool]:
    """Per-head attention of the query at `q_pos`. Returns arrays and whether rows were cut."""
    n_layers, n_heads = len(attentions), attentions[0].shape[1]
    if subj_idx:
        heads = torch.stack([a[0, :, q_pos, subj_idx].float().sum(-1) for a in attentions])  # [L, H]
        heads_np = heads.cpu().numpy().astype(np.float32)
    else:
        heads_np = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    rows = np.full((len(row_layers), n_heads, max_len), np.nan, dtype=np.float16)
    k = min(q_pos + 1, max_len)  # causal: keys 0..q_pos
    for j, layer in enumerate(row_layers):
        rows[j, :, :k] = attentions[layer][0, :, q_pos, :k].float().cpu().numpy()
    return {"attn_subj_heads_tbg": heads_np, "attn_row_tbg": rows}, q_pos + 1 > max_len


# ----------------------------------------------------------------------------- per question

@torch.no_grad()
def greedy_pass(model: Any, input_ids: torch.Tensor, gcfg: dict[str, Any], eos_ids: set[int]) -> dict[str, Any]:
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        max_new_tokens=gcfg["max_new_tokens"],
        return_dict_in_generate=True,
        output_logits=True,
    )
    P = input_ids.shape[1]
    tokens, ends = truncate_at_eos(out.sequences[0, P:].tolist(), eos_ids)
    logits = torch.stack(out.logits[: len(tokens)], 0)[:, 0, :].float()
    logp = F.log_softmax(logits, dim=-1)
    tgt = torch.tensor(tokens, device=logp.device)[:, None]
    return {
        "token_ids": tokens,
        "ends_with_eos": ends,
        "token_entropy": (-(logp.exp() * logp).sum(-1)).tolist(),
        "token_logprob": logp.gather(1, tgt).squeeze(1).tolist(),
    }


@torch.no_grad()
def sample_pass(
    model: Any, tok: Any, input_ids: torch.Tensor, gcfg: dict[str, Any], eos_ids: set[int], seed: int
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=True,
        temperature=gcfg["sample_temperature"],
        top_p=gcfg.get("sample_top_p", 1.0),
        top_k=gcfg.get("sample_top_k", 0),
        num_return_sequences=gcfg["n_samples"],
        max_new_tokens=gcfg["max_new_tokens"],
        return_dict_in_generate=True,
        output_logits=True,
    )
    seqs = out.sequences[:, input_ids.shape[1]:]
    # Log-probs under the model's own distribution (raw logits, T=1), step by step to save memory.
    lp = torch.stack(
        [F.log_softmax(lg.float(), -1).gather(1, seqs[:, t : t + 1]).squeeze(1) for t, lg in enumerate(out.logits)],
        dim=1,
    )
    samples = []
    for n in range(seqs.shape[0]):
        tokens, ends = truncate_at_eos(seqs[n].tolist(), eos_ids)
        samples.append(
            {
                "text": decode_answer(tok, tokens, ends),
                "token_ids": tokens,
                "token_logprob": lp[n, : len(tokens)].tolist(),
                "ends_with_eos": ends,
            }
        )
    return samples


@torch.no_grad()
def process_question(
    model: Any, tok: Any, q: dict[str, Any], gcfg: dict[str, Any], eos_ids: set[int],
    layers: list[int], seed: int, row_layers: list[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = model.device
    prompt_ids, subj_idx = encode_prompt(tok, q["question"], q.get("subject"), gcfg.get("system_prompt"))
    input_ids = torch.tensor([prompt_ids], device=device)
    P = len(prompt_ids)

    greedy = greedy_pass(model, input_ids, gcfg, eos_ids)
    greedy["text"] = decode_answer(tok, greedy["token_ids"], greedy["ends_with_eos"])

    # One forward pass over prompt + greedy answer for hidden states and attentions.
    full = torch.tensor([prompt_ids + greedy["token_ids"]], device=device)
    fo = model(input_ids=full, output_hidden_states=True, output_attentions=True, use_cache=False)
    hs = fo.hidden_states
    S = full.shape[1]
    tbg, slt = P - 1, S - 2  # T >= 1 always, so slt >= tbg and slt + 1 < S

    norm, head = final_norm_and_head(model)
    arrays: dict[str, np.ndarray] = {
        "hidden_tbg": torch.stack([hs[i][0, tbg] for i in layers]).cpu().numpy().astype(np.float16),
        "hidden_slt": torch.stack([hs[i][0, slt] for i in layers]).cpu().numpy().astype(np.float16),
    }
    for name, pos in (("tbg", tbg), ("slt", slt)):
        for k, v in logit_lens(hs, pos, int(full[0, pos + 1]), norm, head).items():
            arrays[f"lens_{k}_{name}"] = v
    attn = subject_attention(
        fo.attentions, subj_idx,
        {"tbg": [tbg], "slt": [slt], "answer_mean": list(range(P - 1, S - 1))},
    )
    for k, v in attn.items():
        arrays[f"attn_subj_{k}"] = v
    per_head, row_cut = tbg_attention_per_head(
        fo.attentions, subj_idx, tbg, row_layers, gcfg.get("attn_row_max_len", 128)
    )
    arrays.update(per_head)
    del fo, hs

    samples = sample_pass(model, tok, input_ids, gcfg, eos_ids, seed)

    record = {
        "qid": q["qid"],
        "split": q["split"],
        "question": q["question"],
        "subject": q.get("subject"),
        "aliases": q["aliases"],
        "prompt_len": P,
        "positions": {"tbg": tbg, "slt": slt},
        "subject_token_idx": subj_idx,
        "attn_row_truncated": row_cut,
        "greedy": greedy,
        "samples": samples,
    }
    return record, arrays


# ----------------------------------------------------------------------------- stage

def _stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def finalize(
    store: ShardStore, questions: list[dict[str, Any]], out: Path, checkpoint: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for c in range(store.n_chunks(len(questions))):
        recs, arrs = store.load_chunk(c)
        records.extend(recs)
        for k, v in arrs.items():
            parts[k].append(v)
    if [r["qid"] for r in records] != [q["qid"] for q in questions]:
        raise RuntimeError("merged shards do not match questions.jsonl order")

    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    write_jsonl(out / "generations.jsonl", records)
    for k in HIDDEN_KEYS:
        np.save(out / f"{k}.npy", arrays.pop(k))
    np.savez(out / "features.npz", **arrays)

    n = len(records)
    secs = [r["seconds"] for r in records if r.get("seconds") is not None]  # absent in M1 shards
    samples = [s for r in records for s in r["samples"]]
    stats = {
        "n_questions": n,
        "n_expected": len(questions),
        "greedy_ends_with_eos": sum(r["greedy"]["ends_with_eos"] for r in records) / n,
        "greedy_empty": sum(not r["greedy"]["text"] for r in records),
        "subject_found": sum(bool(r["subject_token_idx"]) for r in records) / n,
        "mean_greedy_tokens": float(np.mean([len(r["greedy"]["token_ids"]) for r in records])),
        "n_samples": len(samples),
        "samples_ends_with_eos": sum(s["ends_with_eos"] for s in samples) / len(samples) if samples else None,
        "mean_sample_tokens": float(np.mean([len(s["token_ids"]) for s in samples])) if samples else None,
        "seconds_per_question": float(np.mean(secs)) if secs else None,
        "attn_row_truncated": sum(bool(r.get("attn_row_truncated")) for r in records),
        "n_timed": len(secs),
        "checkpoint": checkpoint,
    }
    write_json(out / "summary.json", stats)
    return stats, records


def fill_report(part: Any, stats: dict[str, Any], records: list[dict[str, Any]], leak_ok: bool) -> None:
    n = stats["n_questions"]
    ck = stats["checkpoint"]
    eos = round(stats["greedy_ends_with_eos"] * n)
    subj = round(stats["subject_found"] * n)
    spq = stats["seconds_per_question"]
    part.n = n
    part.summary = stats
    part.hard(ck["complete"], f"all checkpoints complete ({ck['done']}/{ck['total']} chunks)"
              if ck["complete"] else f"only {ck['done']} of {ck['total']} checkpoint chunks finished")
    part.hard(leak_ok, "no subject leakage across splits" if leak_ok else "a subject appears in two splits")
    part.soft(stats["greedy_empty"] == 0, "no empty predictions" if not stats["greedy_empty"]
              else f"{stats['greedy_empty']} greedy answers are empty")
    part.soft(stats["greedy_ends_with_eos"] >= 0.95, f"{100 * stats['greedy_ends_with_eos']:.1f}% of greedy answers "
              "finished cleanly" + ("" if stats["greedy_ends_with_eos"] >= 0.95 else " (expected ≥ 95%; raise max_new_tokens?)"))
    part.soft(stats["subject_found"] >= 0.95, f"subject located in the prompt for {100 * stats['subject_found']:.1f}% of questions"
              + ("" if stats["subject_found"] >= 0.95 else " (expected ≥ 95%; attention features will be NaN for the rest)"))

    t = f"{spq:.2f} s/question" if spq else "time n/a (older shards)"
    part.key.append(f"questions done {n} / {stats['n_expected']} | {t} | "
                    f"mean answer tokens {stats['mean_greedy_tokens']:.1f}")
    part.key.append(f"finished cleanly {pct(eos, n)} | empty {stats['greedy_empty']} | subject found {pct(subj, n)}")
    part.claude.append(
        f"n={n}/{stats['n_expected']} s/q={spq if spq is None else round(spq, 3)} "
        f"tok_greedy={stats['mean_greedy_tokens']:.2f} tok_samples={stats['mean_sample_tokens']} "
        f"eos_greedy={stats['greedy_ends_with_eos']:.4f} eos_samples={stats['samples_ends_with_eos']} "
        f"empty={stats['greedy_empty']} subj_found={stats['subject_found']:.4f} "
        f"attn_rows_cut={stats['attn_row_truncated']} checkpoints={ck}"
    )
    part.plots.append(("Greedy answer length", histogram([len(r["greedy"]["token_ids"]) for r in records], "tokens")))


def run(cfg: dict[str, Any], report: Any = None) -> None:
    gcfg, mcfg, seed = cfg["generation"], cfg["model"], cfg["seed"]
    rd = run_dir(cfg)
    out = rd / "generation"
    questions = read_jsonl(rd / "data" / "questions.jsonl")

    fp = fingerprint(
        {
            "model": mcfg,
            "generation": {k: v for k, v in gcfg.items() if k != "resume_dirs"},
            "seed": seed,
            "qids": [q["qid"] for q in questions],
        }
    )
    store = ShardStore(out / "shards", gcfg.get("checkpoint_every", 100), fp)
    imported = store.import_from(gcfg["resume_dirs"]) if gcfg.get("resume_dirs") else 0
    if imported:
        print(f"[generation] imported {imported} shards from resume_dirs")

    n_layers = AutoConfig.from_pretrained(mcfg["id"], revision=mcfg.get("revision")).num_hidden_layers
    layers = selected_layers(n_layers, gcfg.get("layer_stride", 4))
    row_layers = attention_row_layers(n_layers, gcfg.get("attn_row_layer_fracs", [0.5, 0.75, 0.9]))
    write_json(out / "layers.json", {
        "hidden_state_indices": layers, "num_hidden_layers": n_layers,
        "attn_row_layers": row_layers, "attn_row_max_len": gcfg.get("attn_row_max_len", 128),
    })

    todo = store.missing_chunks(len(questions))
    if todo:
        set_seed(seed)
        model, tok, eos_ids = load_model(mcfg)
        print(f"[generation] {len(todo)} chunks to do; caching hidden-state indices {layers}")
        for c in todo:
            recs, rows = [], []
            for i in tqdm(store.chunk_range(c, len(questions)), desc=f"chunk {c}"):
                t0 = time.perf_counter()
                rec, arr = process_question(model, tok, questions[i], gcfg, eos_ids, layers, seed + i, row_layers)
                rec["seconds"] = round(time.perf_counter() - t0, 3)
                recs.append(rec)
                rows.append(arr)
            store.save_chunk(c, recs, _stack(rows))
            print(f"[generation] saved chunk {c}")
        del model
        torch.cuda.empty_cache()
    else:
        print("[generation] all chunks already done; re-merging")

    total = store.n_chunks(len(questions))
    missing = store.missing_chunks(len(questions))
    checkpoint = {
        "total": total, "done": total - len(missing), "complete": not missing,
        "chunk_size": store.chunk_size, "generated": len(todo), "imported": imported,
    }
    if missing:
        raise RuntimeError(f"checkpoint chunks {missing} are missing after generation")
    stats, records = finalize(store, questions, out, checkpoint)
    write_meta(out, cfg, "generation", {"fingerprint": fp, "summary": stats})
    try:
        check_no_leakage(questions)
        leak_ok = True
    except AssertionError:
        leak_ok = False
    if report is not None:
        fill_report(report, stats, records, leak_ok)
    print(f"[generation] done -> {out}")
