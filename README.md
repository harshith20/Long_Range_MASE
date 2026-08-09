# Edge-MASE

Edge-MASE is a lightweight hallucination and self-contradiction detection framework for Small Language Models (SLMs) on edge devices. It extends the Multi-scale Adaptive Semantic Entropy (MASE) framework with a Top-k Sparse NLI Adjacency Graph to detect long-range contradictions efficiently.

## Overview

The system computes a scalar risk score for each newly generated sentence using three complementary signals:

- SEG: Semantic Entropy Gradient — captures sudden spikes in token-level uncertainty.
- H_adjacent: Adjacent transition entropy — measures incoherence between consecutive sentences using NLI outputs.
- C_long_range: Maximum contradiction probability from a top-k semantic retrieval + NLI check against past sentences.

Mathematical formulation:

Risk = w1 * SEG + w2 * H_adjacent + w3 * C_long_range

where w1, w2, w3 are tunable weights.

If Risk exceeds a configured threshold an alarm is raised (log, callback, or downstream mitigation).

## Components

- `src/seg_tracker.py` — tracks token probabilities and computes SEG over a sliding window.
- `src/adjacent_coherence.py` — computes Shannon transition entropy between consecutive sentences using `roberta-large-mnli`.
- `src/topk_graph.py` — maintains an in-memory FAISS index over sentence embeddings (`all-MiniLM-L6-v2`) and runs NLI checks against top-k nearest past sentences to compute long-range contradiction.
- `src/mase_engine.py` — coordinator that ingests sentences, calls the submodules, computes the risk score, and triggers alarms.

## Quick start (local)

1. Create a Python environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run a small script or the notebook to instantiate the `MASEEngine` and pass sentences into `process_sentence()`.

## Running on Google Colab

The included notebook `notebooks/edge_slm_evaluation.ipynb` contains Colab-ready cells to install dependencies and demonstrate a sentence-by-sentence risk evaluation loop. For very large LLMs you can load quantized models with `bitsandbytes` and accelerate with a GPU runtime.

## Notes on models and privacy

- The code uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `roberta-large-mnli` for NLI. On-edge deployments should prefer quantized, distilled, or smaller NLI models to reduce footprint.
