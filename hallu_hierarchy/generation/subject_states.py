"""Stage `subject_states` (H4): one prompt-only forward pass per question.

Caches the hidden state at the subject's LAST token for the same layers as generation
(every `layer_stride`-th hidden-state index + the last), fp16. These states feed the
subject-familiarity probe (M4): does the model "know" the subject, independent of the answer?

Consistency with the generation stage (asserted, CLAUDE.md §3):
  * the prompt length and subject token positions must equal those saved in generations.jsonl
    for every question (same chat template, same tokenizer);
  * on a sample, the prompt-only state at the last prompt token must equal the saved `hidden_tbg`
    (causal attention: the answer tokens cannot change earlier positions) up to fp16 noise.

Outputs (outputs/<run>/subject_states/):
  hidden_subject.npy   [N, n_layers_cached, d] fp16, NaN rows where the subject was not found
  index.jsonl          qid, subject_found, subject_last_idx, prompt_len (row order = questions.jsonl)
  layers.json, summary.json, meta.json, shards/ (checkpoints)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import find_upstream, read_jsonl, run_dir, set_seed, write_json, write_jsonl, write_meta

TBG_CHECK_SAMPLE = 20
TBG_TOLERANCE = 0.05  # max |diff| / max |value| between the two fp16 TBG states


# ----------------------------------------------------------------------------- pure helpers

def subject_last_index(subj_idx: list[int]) -> int | None:
    """Position of the subject's last token, or None if the subject was not located."""
    return max(subj_idx) if subj_idx else None


def compare_with_generation(ours: list[dict[str, Any]], gen: dict[str, dict[str, Any]]) -> list[str]:
    """Mismatches in prompt length / subject positions vs the generation stage (empty list = identical)."""
    bad = []
    for r in ours:
        g = gen.get(r["qid"])
        if g is None:
            bad.append(f"{r['qid']}: not in generations.jsonl")
        elif g["prompt_len"] != r["prompt_len"] or list(g["subject_token_idx"]) != list(r["subject_token_idx"]):
            bad.append(f"{r['qid']}: prompt_len {g['prompt_len']}→{r['prompt_len']}, "
                       f"subject {g['subject_token_idx']}→{r['subject_token_idx']}")
    return bad


def relative_diff(a: np.ndarray, b: np.ndarray) -> float:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    scale = float(np.max(np.abs(b32))) or 1.0
    return float(np.max(np.abs(a32 - b32))) / scale


# ----------------------------------------------------------------------------- model part

def prompt_states(model: Any, prompt_ids: list[int], positions: list[int], layers: list[int]) -> list[np.ndarray]:
    """Hidden states [len(layers), d] (fp16) at each position, from one prompt-only forward pass."""
    import torch

    with torch.no_grad():
        ids = torch.tensor([prompt_ids], device=model.device)
        hs = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True,
                   use_cache=False).hidden_states
        return [torch.stack([hs[i][0, p] for i in layers]).float().cpu().numpy().astype(np.float16)
                for p in positions]


def process_question(model: Any, tok: Any, q: dict[str, Any], system_prompt: str | None,
                     layers: list[int], d: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from .prompts import encode_prompt

    prompt_ids, subj_idx = encode_prompt(tok, q["question"], q.get("subject"), system_prompt)
    last = subject_last_index(subj_idx)
    tbg = len(prompt_ids) - 1
    if last is None:
        (tbg_state,) = prompt_states(model, prompt_ids, [tbg], layers)
        subj_state = np.full((len(layers), d), np.nan, dtype=np.float16)
    else:
        subj_state, tbg_state = prompt_states(model, prompt_ids, [last, tbg], layers)
    rec = {"qid": q["qid"], "prompt_len": len(prompt_ids), "subject_token_idx": subj_idx,
           "subject_last_idx": last, "subject_found": last is not None}
    return rec, {"hidden_subject": subj_state, "hidden_tbg_check": tbg_state}


# ----------------------------------------------------------------------------- stage

def run(cfg: dict[str, Any], report: Any = None) -> None:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoConfig

    from .checkpoint import ShardStore, fingerprint
    from .generate import load_model, selected_layers

    gcfg, mcfg, seed = cfg["generation"], cfg["model"], cfg["seed"]
    out = run_dir(cfg) / "subject_states"
    q_path = find_upstream(cfg, "data/questions.jsonl")
    gen_path = find_upstream(cfg, "generation/generations.jsonl")
    if q_path is None or gen_path is None:
        raise FileNotFoundError("subject_states needs data/questions.jsonl and generation/generations.jsonl "
                                "(attach the dataset with the saved generation outputs)")
    questions = read_jsonl(q_path)
    gen_records = {r["qid"]: r for r in read_jsonl(gen_path)}

    n_layers = AutoConfig.from_pretrained(mcfg["id"], revision=mcfg.get("revision")).num_hidden_layers
    layers = selected_layers(n_layers, gcfg.get("layer_stride", 4))
    gen_layers_path = gen_path.parent / "layers.json"
    if gen_layers_path.exists():
        import json
        gen_layers = json.loads(gen_layers_path.read_text())["hidden_state_indices"]
        if gen_layers != layers:
            raise RuntimeError(f"layers {layers} differ from the generation stage {gen_layers}")
    write_json(out / "layers.json", {"hidden_state_indices": layers, "num_hidden_layers": n_layers,
                                     "position": "subject last token (prompt-only pass)"})

    fp = fingerprint({"stage": "subject_states", "model": mcfg, "system_prompt": gcfg.get("system_prompt"),
                      "layers": layers, "qids": [q["qid"] for q in questions]})
    store = ShardStore(out / "shards", gcfg.get("checkpoint_every", 100), fp)
    imported = store.import_from(cfg.get("subject_states", {}).get("resume_dirs", []) or [])
    todo = store.missing_chunks(len(questions))
    if todo:
        set_seed(seed)
        model, tok, _ = load_model(mcfg)
        d = model.config.hidden_size
        print(f"[subject_states] {len(todo)} chunks to do; hidden-state indices {layers}")
        for c in todo:
            recs, rows = [], []
            for i in tqdm(store.chunk_range(c, len(questions)), desc=f"chunk {c}", mininterval=30):
                t0 = time.perf_counter()
                rec, arr = process_question(model, tok, questions[i], gcfg.get("system_prompt"), layers, d)
                rec["seconds"] = round(time.perf_counter() - t0, 4)
                recs.append(rec)
                rows.append(arr)
            store.save_chunk(c, recs, {k: np.stack([r[k] for r in rows]) for k in rows[0]})
        del model
        torch.cuda.empty_cache()
    missing = store.missing_chunks(len(questions))
    if missing:
        raise RuntimeError(f"checkpoint chunks {missing} are missing")

    # merge
    records, subj, tbg = [], [], []
    for c in range(store.n_chunks(len(questions))):
        recs, arrs = store.load_chunk(c)
        records += recs
        subj.append(arrs["hidden_subject"])
        tbg.append(arrs["hidden_tbg_check"])
    if [r["qid"] for r in records] != [q["qid"] for q in questions]:
        raise RuntimeError("merged shards do not match questions.jsonl order")
    np.save(out / "hidden_subject.npy", np.concatenate(subj))
    write_jsonl(out / "index.jsonl", [{k: r[k] for k in ("qid", "subject_found", "subject_last_idx", "prompt_len")}
                                      for r in records])

    # consistency with the generation stage
    mismatches = compare_with_generation(records, gen_records)
    tbg_all = np.concatenate(tbg)
    tbg_diffs: list[float] = []
    gen_tbg_path = gen_path.parent / "hidden_tbg.npy"
    if gen_tbg_path.exists():
        gen_tbg = np.load(gen_tbg_path, mmap_mode="r")
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(records), size=min(TBG_CHECK_SAMPLE, len(records)), replace=False)
        tbg_diffs = [relative_diff(tbg_all[i], np.asarray(gen_tbg[i])) for i in sorted(sample)]

    n = len(records)
    secs = [r["seconds"] for r in records if r.get("seconds") is not None]
    found = sum(r["subject_found"] for r in records)
    stats = {
        "n_questions": n, "subject_found": found / n if n else 0.0, "n_subject_missing": n - found,
        "seconds_per_question": float(np.mean(secs)) if secs else None, "n_timed": len(secs),
        "prompt_mismatches": len(mismatches), "prompt_mismatch_examples": mismatches[:5],
        "tbg_check_n": len(tbg_diffs), "tbg_max_rel_diff": max(tbg_diffs) if tbg_diffs else None,
        "checkpoint": {"total": store.n_chunks(n), "generated": len(todo), "imported": imported},
        "layers": layers, "shape": list(np.load(out / "hidden_subject.npy", mmap_mode="r").shape),
    }
    write_json(out / "summary.json", stats)
    write_meta(out, cfg, "subject_states", {"fingerprint": fp, "summary": stats})
    if report is not None:
        fill_report(report, stats)
    print(f"[subject_states] done -> {out}")


def fill_report(part: Any, s: dict[str, Any]) -> None:
    from ..reporting.console import pct

    n = s["n_questions"]
    part.n = n
    part.summary = s
    ck = s["checkpoint"]
    part.hard(True, f"all checkpoints complete ({ck['total']}/{ck['total']} chunks)")
    part.hard(s["prompt_mismatches"] == 0,
              "prompts and subject positions identical to the generation stage (all questions)"
              if s["prompt_mismatches"] == 0 else
              f"{s['prompt_mismatches']} questions have a different prompt/subject position than in generation, "
              f"e.g. {s['prompt_mismatch_examples'][:2]}")
    if s["tbg_max_rel_diff"] is None:
        part.soft(False, "saved hidden_tbg.npy not found: last-prompt-token state not cross-checked")
    else:
        ok = s["tbg_max_rel_diff"] <= TBG_TOLERANCE
        part.hard(ok, f"last-prompt-token state matches generation on {s['tbg_check_n']} questions "
                      f"(max relative diff {s['tbg_max_rel_diff']:.4f})" if ok else
                      f"last-prompt-token state differs from generation (max relative diff "
                      f"{s['tbg_max_rel_diff']:.4f} > {TBG_TOLERANCE}): prompts or model are not the same")
    sf = s["subject_found"]
    part.soft(sf >= 0.95, f"subject located for {100 * sf:.1f}% of questions"
              + ("" if sf >= 0.95 else " (expected ≥ 95%; subject-state features will be missing for the rest)"))
    spq = s["seconds_per_question"]
    part.key.append(f"questions done {n} | {spq:.3f} s/question" if spq else f"questions done {n}")
    part.key.append(f"subject found {pct(n - s['n_subject_missing'], n)} | states {s['shape']} fp16 "
                    f"at hidden-state indices {s['layers']}")
    part.claude.append(f"n={n} s/q={spq} subject_found={sf:.4f} missing={s['n_subject_missing']} "
                       f"prompt_mismatches={s['prompt_mismatches']} tbg_max_rel_diff={s['tbg_max_rel_diff']} "
                       f"shape={s['shape']} checkpoints={ck}")
