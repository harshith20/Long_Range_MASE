#!/usr/bin/env python3
"""Evaluate MASE variants on the pminervini/HaluEval dataset (static text-level).

This script intentionally disables the SEG/token-level term (w1=0.0)
and forces all model components to CPU to run on non-GPU machines.
"""

import time
import sys
from typing import List, Dict, Any

import pandas as pd
from datasets import load_dataset

import nltk
from nltk import sent_tokenize

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

# Import the extended wrapper from the local package
try:
    from src.mase_engine import ExtendedMASEEngine
except Exception as e:
    print("Error importing ExtendedMASEEngine from src.mase_engine:", e, file=sys.stderr)
    raise


def run_evaluation_pass(dataset_samples: List[Dict[str, Any]], w2: float, w4: float, threshold: float, config_name: str) -> Dict[str, Any]:
    """Run one evaluation pass over a list of dataset samples.

    Args:
        dataset_samples: list-like of dataset dicts (each must contain 'hallucinated_summary' or similar)
        w2: weight for adjacent NLI
        w4: weight for long-range/top-k NLI (mapped to internal w3)
        threshold: alarm threshold
        config_name: descriptive name for logging

    Returns:
        dict with Total Samples, Alarms Triggered, Recall (%), Avg Latency per Sentence (ms)
    """

    total_samples = 0
    alarms = 0
    sentence_latencies_ms: List[float] = []

    for sample in dataset_samples:
        total_samples += 1
        # Re-instantiate engine per-sample as required; force CPU
        try:
            engine = ExtendedMASEEngine(w1=0.0, w2=w2, w4=w4, threshold=threshold, device="cpu")
        except Exception as e:
            print(f"[{config_name}] Failed to init engine for sample #{total_samples}: {e}", file=sys.stderr)
            continue

        # Extract hallucinated summary field directly from the dataset sample.
        try:
            text = sample["hallucinated_summary"]
        except Exception:
            text = None

        if not text:
            print(f"[{config_name}] Sample #{total_samples} missing expected summary field; skipping.", file=sys.stderr)
            continue

        # Split into sentences
        try:
            sentences = sent_tokenize(str(text))
        except Exception as e:
            print(f"[{config_name}] Error tokenizing sample #{total_samples}: {e}", file=sys.stderr)
            continue

        triggered = False
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
                triggered = True
                break  # stop processing this sample after alarm

        # Optional: if no sentences produced, continue
        if not sentences:
            continue

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
    print("Loading dataset (pminervini/HaluEval) -- this may download models/dataset.")
    try:
        dataset = load_dataset("pminervini/HaluEval", "summarization", split="data[:50]")
    except Exception as e:
        print("Failed to load dataset:", e, file=sys.stderr)
        sys.exit(1)
    samples = list(dataset)
    print(f"Loaded {len(samples)} samples for evaluation.")

    # Configuration A: Baseline MASE (Adjacent NLI only)
    print("Running Configuration A (Baseline MASE): w1=0.0, w2=1.0, w4=0.0, threshold=0.70")
    res_a = run_evaluation_pass(samples, w2=1.0, w4=0.0, threshold=0.70, config_name="Config A")

    # Configuration B: Extended MASE (Adjacent + Top-k FAISS Graph)
    print("Running Configuration B (Extended MASE): w1=0.0, w2=0.5, w4=0.5, threshold=0.70")
    res_b = run_evaluation_pass(samples, w2=0.5, w4=0.5, threshold=0.70, config_name="Config B")

    # Comparative summary
    df = pd.DataFrame([
        {"Config": res_a["config"], "Recall (%)": round(res_a["recall_pct"], 2), "Avg Latency (ms)": round(res_a["avg_latency_ms"], 2)},
        {"Config": res_b["config"], "Recall (%)": round(res_b["recall_pct"], 2), "Avg Latency (ms)": round(res_b["avg_latency_ms"], 2)},
    ])

    print("\nComparative Evaluation Summary:")
    print(df.to_string(index=False))

    # Also return non-zero exit code if both configs failed to run
    if res_a["total_samples"] == 0 and res_b["total_samples"] == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
