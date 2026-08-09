This repository implements a Top-k sparse semantic retrieval + NLI contradiction pipeline.

Notes for GitHub Copilot:
- Use FAISS (faiss-cpu) to maintain a small in-memory IndexFlatIP index for `all-MiniLM-L6-v2` embeddings.
- Use `sentence-transformers` to produce embeddings and normalize them for cosine similarity.
- For contradiction checks, prefer `roberta-large-mnli` (via `transformers` pipeline 'text-classification' with `return_all_scores=True`).
- Keep on-edge implementations lightweight: allow fallbacks when models are not available, and expose hooks for quantized or distilled NLI models.
