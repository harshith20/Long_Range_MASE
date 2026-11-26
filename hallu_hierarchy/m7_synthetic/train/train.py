"""M7: train a small decoder from scratch on the synthetic corpus (plan §2, decisions D5, D6).

One pass over the token shards in their stored order (D5: exposure = sentence count; no reshuffling).
Architecture: GPT-NeoX-style with rotary positions (Allen-Zhu Part 3.1 replace GPT-2's learned positions),
GPT-2-small body (12 layers, 12 heads, 768 dims). Optimizer as in Allen-Zhu Part 3.1 App. C:
AdamW (wd 0.1, eps 1e-6), lr 1e-3 -> 1e-4 cosine, linear warmup.

Usage:
    python -m hallu_hierarchy.m7_synthetic.train.train --data <corpus dir> --out <run dir> --model s124
    torchrun --nproc_per_node=2 -m hallu_hierarchy.m7_synthetic.train.train ...      (2 x T4)
Resumes automatically from <out>/state.pt if present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELS = {
    # GPT-2-small body with rotary positions (~85M non-embedding parameters)
    "s124": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "s25": dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048),
    "tiny": dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=512),
}
DEFAULT_CKPT_FRACS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]  # denser late (Zucchet et al.)


def build_model(name: str, vocab_size: int, eos_id: int, seq_len: int):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    vocab = int(math.ceil(vocab_size / 64) * 64)
    cfg = GPTNeoXConfig(vocab_size=vocab, max_position_embeddings=seq_len, rotary_pct=0.25,
                        rotary_emb_base=10000, use_parallel_residual=False, hidden_act="gelu",
                        bos_token_id=eos_id, eos_token_id=eos_id, tie_word_embeddings=False,
                        attention_dropout=0.0, hidden_dropout=0.0, **MODELS[name])
    try:
        return GPTNeoXForCausalLM._from_config(cfg, attn_implementation="sdpa")
    except (TypeError, ValueError):
        return GPTNeoXForCausalLM(cfg)


class TokenStream:
    """Concatenated uint16 shards read in stored order, cut into non-overlapping windows of seq_len + 1."""

    def __init__(self, data_dir: Path, seq_len: int):
        shards = sorted(data_dir.glob("tokens_*.bin"))
        if not shards:
            raise FileNotFoundError(f"no tokens_*.bin in {data_dir}")
        self.arr = np.concatenate([np.memmap(s, dtype=np.uint16, mode="r") for s in shards]) \
            if len(shards) > 1 else np.memmap(shards[0], dtype=np.uint16, mode="r")
        self.seq = seq_len
        self.n_windows = (len(self.arr) - 1) // seq_len

    def window(self, i: int) -> np.ndarray:
        s = i * self.seq
        return np.asarray(self.arr[s:s + self.seq + 1], dtype=np.int64)


def lr_at(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, p)))


def quick_eval(model, tok, people, device, n: int = 500) -> dict:
    """First-token accuracy on qa_test birth city: QA prompt and completion prompt (no QA needed)."""
    test = [p for p in people if p["split"] == "qa_test"]
    rng = random.Random(0)
    sample = rng.sample(test, min(n, len(test)))
    out = {}
    for kind, fmt in (("qa", "Q: Where was {name} born?\nA:"), ("completion", "{name} was born in")):
        prompts = [fmt.format(name=p["name"]) for p in sample]
        gold = [tok.encode(" " + p["birth_city"]).ids[0] for p in sample]
        correct = 0
        for b in range(0, len(prompts), 64):
            encs = [tok.encode(t).ids for t in prompts[b:b + 64]]
            L = max(map(len, encs))
            ids = torch.zeros((len(encs), L), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, e in enumerate(encs):
                ids[i, :len(e)] = torch.tensor(e)
                mask[i, :len(e)] = 1
            with torch.no_grad():
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            last = mask.sum(1) - 1
            pred = logits[torch.arange(len(encs)), last.to(device)].argmax(-1).cpu().tolist()
            correct += sum(int(a == g) for a, g in zip(pred, gold[b:b + 64]))
        out[f"{kind}_acc"] = correct / len(sample)
    # by exposure level (completion prompt only would need another pass; keep it cheap)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="s124", choices=list(MODELS))
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32, help="global batch (sequences)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--ckpt-fracs", default=",".join(map(str, DEFAULT_CKPT_FRACS)))
    ap.add_argument("--time-limit-h", type=float, default=11.0, help="save state and exit before this")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args(argv)

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = 0, 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")  # gloo: local CPU test
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = json.loads((data_dir / "budget.json").read_text())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    import pandas as pd
    people = pd.read_parquet(data_dir / "people.parquet").to_dict("records")

    stream = TokenStream(data_dir, args.seq)
    total = stream.n_windows // args.batch
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = min(args.warmup, max(1, total // 20))
    ckpt_steps = sorted({max(1, int(round(f * total))) for f in map(float, args.ckpt_fracs.split(","))})
    if args.batch % world:
        raise ValueError("global batch must be divisible by the number of GPUs")
    per_rank = args.batch // world

    model = build_model(args.model, tok.get_vocab_size(), budget["eos_id"], args.seq).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_body = n_params - sum(p.numel() for n, p in model.named_parameters() if "embed" in n)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=device.type == "cuda")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step = 0
    state_path = out / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step = st["step"]
        if main_proc:
            print(f"resumed at step {step}/{total}")
    train_model = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    meta = {"model": args.model, "params": n_params, "non_embedding_params": n_body, "steps": total,
            "batch": args.batch, "seq": args.seq, "tokens_per_step": args.batch * args.seq, "warmup": warmup,
            "lr": args.lr, "min_lr": args.min_lr, "wd": args.wd, "seed": args.seed, "world_size": world,
            "amp": str(amp_dtype) if use_amp else "fp32", "ckpt_steps": ckpt_steps,
            "corpus_tokens": int(len(stream.arr)), "generator_version": budget.get("generator_version"),
            "data_seed": budget.get("seed")}
    if main_proc:
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))
    log_f = open(out / "train_log.jsonl", "a") if main_proc else None

    t0 = time.time()
    t_last, tok_last = time.time(), 0
    model.train()
    while step < total:
        rows = range(step * args.batch + rank * per_rank, step * args.batch + (rank + 1) * per_rank)
        batch = torch.from_numpy(np.stack([stream.window(i) for i in rows])).to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, warmup, args.lr, args.min_lr)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = train_model(input_ids=x).logits
            loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tok_last += args.batch * args.seq
        if main_proc and (step % args.log_every == 0 or step == total):
            dt = time.time() - t_last
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": opt.param_groups[0]["lr"],
                   "tok_per_s": round(tok_last / dt), "elapsed_h": round((time.time() - t0) / 3600, 3)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(rec, flush=True)
            t_last, tok_last = time.time(), 0
        save_ckpt = step in ckpt_steps
        out_of_time = False
        if step % args.log_every == 0 and step < total:  # all ranks decide together (else DDP hangs)
            flag = torch.tensor([float((time.time() - t0) / 3600 > args.time_limit_h)], device=device)
            if ddp:
                import torch.distributed as dist
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            out_of_time = bool(flag.item())
        if main_proc and (save_ckpt or out_of_time):
            if save_ckpt:
                model.eval()
                qe = quick_eval(model, tok, people, device)
                model.train()
                d = out / f"ckpt_{step:06d}"
                model.save_pretrained(d, safe_serialization=True)
                rec = {"checkpoint": d.name, "step": step, "tokens_seen": step * args.batch * args.seq, **qe}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(rec, flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "step": step}, state_path)
        if out_of_time:
            if main_proc:
                print(f"time limit reached at step {step}; state saved, rerun to resume", flush=True)
            break
    if main_proc:
        (out / "DONE" if step >= total else out / "PAUSED").write_text(str(step))
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
