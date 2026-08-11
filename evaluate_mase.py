#!/usr/bin/env python3
"""Evaluate MASE variants on the pminervini/HaluEval dataset (static text-level)."""

import time
import sys
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset

import nltk
from nltk import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

try:
    from src.mase_engine import ExtendedMASEEngine
except Exception as e:
    print("Error importing ExtendedMASEEngine from src.mase_engine:", e, file=sys.stderr)
    raise


def run_evaluation_pass(dataset_samples: List[Dict[str, Any]], w2: float, w4: float, threshold: float, config_name: str) -> Dict[str, Any]:
    total_samples = 0
    alarms = 0
    sentence_latencies_ms: List[float] = []

    print(f"\n[{config_name}] Initializing models on CPU...")
    try:
        engine = ExtendedMASEEngine(w1=0.0, w2=w2, w4=w4, threshold=threshold, device="cpu")
    except Exception as e:
        print(f"[{config_name}] Failed to init engine: {e}", file=sys.stderr)
        return {"config": config_name, "total_samples": 0, "alarms": 0, "recall_pct": 0.0, "avg_latency_ms": 0.0}

    print(f"[{config_name}] Running evaluation over {len(dataset_samples)} samples...")
    for sample in dataset_samples:
        total_samples += 1
        engine.reset()  # Reset sentence history per document

        try:
            text = sample.get("hallucinated_summary")
        except Exception:
            text = None

        if not text:
            continue

        try:
            sentences = sent_tokenize(str(text))
        except Exception as e:
            print(f"[{config_name}] Error tokenizing sample #{total_samples}: {e}", file=sys.stderr)
            continue

        for sent in sentences:
            t0 = time.perf_counter()
            try:
                report = engine.process_new_sentence(sent)
            except Exception as e:
                print(f"[{config_name}] Error processing sentence for sample #{total_samples}: {e}", file=sys.stderr)
                continue
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            sentence_latencies_ms.append(latency_ms)

            if isinstance(report, dict) and report["alarm"]:
                alarms += 1
                break  # stop processing sample once alarm triggers

    recall_pct = (alarms / total_samples * 100.0) if total_samples > 0 else 0.0
    avg_latency = float(pd.Series(sentence_latencies_ms).mean()) if sentence_latencies_ms else 0.0

    return {
        "config": config_name,
        "total_samples": total_samples,
        "alarms": alarms,
        "recall_pct": recall_pct,
        "avg_latency_ms": avg_latency,
    }


def main():
    print("Loading dataset (pminervini/HaluEval)...")
    try:
        dataset = load_dataset("pminervini/HaluEval", "summarization", split="data[:50]")
    except Exception as e:
        print("Failed to load dataset:", e, file=sys.stderr)
        sys.exit(1)
        
    samples = list(dataset)
    print(f"Loaded {len(samples)} samples for evaluation.")

    # Ensure compatibility across dataset field names
    for s in samples:
        if "hallucinated_summary" not in s and "gpt3_text" in s:
            s["hallucinated_summary"] = s["gpt3_text"]

    # Config A: Baseline MASE (Adjacent NLI only)
    print("\n[Running Config A] Baseline MASE: w2=1.0, w4=0.0, threshold=0.40...")
    res_a = run_evaluation_pass(samples, w2=1.0, w4=0.0, threshold=0.40, config_name="Config A (Baseline)")

    # Config B: Optimized Extended MASE (Adjacent + Top-k FAISS Graph)
    print("[Running Config B] Optimized Extended MASE: w2=0.1, w4=0.9, threshold=0.40...")
    res_b = run_evaluation_pass(samples, w2=0.1, w4=0.9, threshold=0.40, config_name="Config B (Optimized)")

    # Comparative summary
    df = pd.DataFrame([
        {
            "Config": res_a["config"], 
            "Recall (%)": round(res_a["recall_pct"], 2), 
            "Avg Latency (ms)": round(res_a["avg_latency_ms"], 2)
        },
        {
            "Config": res_b["config"], 
            "Recall (%)": round(res_b["recall_pct"], 2), 
            "Avg Latency (ms)": round(res_b["avg_latency_ms"], 2)
        },
    ])

    print("\n==========================================")
    print(" 📊 OPTIMIZED COMPARATIVE EVALUATION SUMMARY")
    print("==========================================")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()