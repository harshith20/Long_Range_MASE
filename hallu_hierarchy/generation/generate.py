"""Stage `generate`: greedy answer + N samples per question, with caching.

Per question we save:
  * greedy answer text/tokens, per-token entropy and log-prob (raw logits, T=1)
  * N sampled answers with per-token log-probs
  * hidden states of the greedy pass at TBG and SLT for a subset of layers (fp16)
  * logit-lens entropy / target log-prob / target-is-top1 at TBG and SLT for every layer
  * attention mass on subject tokens (mean over heads) for every layer

Positions (indices into prompt + greedy answer):
  TBG = last prompt token (state before the first answer token is produced)
  SLT = second-last token of the generated sequence. When the answer ends with EOS this is
        the last real answer token (as in the SEP paper). If only one token was generated,
        SLT coincides with TBG.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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
    layers: list[int], seed: int,
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
        "greedy": greedy,
        "samples": samples,
    }
    return record, arrays


# ----------------------------------------------------------------------------- stage

def _stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def finalize(store: ShardStore, questions: list[dict[str, Any]], out: Path) -> dict[str, Any]:
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
    stats = {
        "n_questions": n,
        "greedy_ends_with_eos": sum(r["greedy"]["ends_with_eos"] for r in records) / n,
        "greedy_empty": sum(not r["greedy"]["text"] for r in records),
        "subject_found": sum(bool(r["subject_token_idx"]) for r in records) / n,
        "mean_greedy_tokens": float(np.mean([len(r["greedy"]["token_ids"]) for r in records])),
    }
    write_json(out / "summary.json", stats)
    return stats


def run(cfg: dict[str, Any]) -> None:
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
    if gcfg.get("resume_dirs"):
        print(f"[generate] imported {store.import_from(gcfg['resume_dirs'])} shards from resume_dirs")

    n_layers = AutoConfig.from_pretrained(mcfg["id"], revision=mcfg.get("revision")).num_hidden_layers
    layers = selected_layers(n_layers, gcfg.get("layer_stride", 4))
    write_json(out / "layers.json", {"hidden_state_indices": layers, "num_hidden_layers": n_layers})

    todo = store.missing_chunks(len(questions))
    if todo:
        set_seed(seed)
        model, tok, eos_ids = load_model(mcfg)
        print(f"[generate] {len(todo)} chunks to do; caching hidden-state indices {layers}")
        for c in todo:
            recs, rows = [], []
            for i in tqdm(store.chunk_range(c, len(questions)), desc=f"chunk {c}"):
                rec, arr = process_question(model, tok, questions[i], gcfg, eos_ids, layers, seed + i)
                recs.append(rec)
                rows.append(arr)
            store.save_chunk(c, recs, _stack(rows))
            print(f"[generate] saved chunk {c}")
        del model
        torch.cuda.empty_cache()
    else:
        print("[generate] all chunks already done; re-merging")

    stats = finalize(store, questions, out)
    write_meta(out, cfg, "generate", {"fingerprint": fp, "summary": stats})
    print(f"[generate] done -> {out}  {stats}")
