from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from transformers import pipeline


class TopKGraph:
    """Maintains a CPU FAISS index of sentence embeddings and runs NLI checks
    against the top-k nearest past sentences to surface contradiction scores.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", nli_model: str = "roberta-large-mnli", k: int = 2) -> None:
        self.k = k
        self.embedder = SentenceTransformer(model_name) if SentenceTransformer is not None else None
        self.dim = self.embedder.get_sentence_embedding_dimension() if self.embedder is not None else 0
        self.index = faiss.IndexFlatIP(self.dim) if (faiss is not None and self.dim > 0) else None
        self.texts: List[str] = []
        self.nli = None
        try:
            self.nli = pipeline("text-classification", model=nli_model, return_all_scores=True)
        except Exception:
            self.nli = None

    def _embed(self, sentence: str) -> np.ndarray:
        if self.embedder is None:
            return np.zeros(self.dim, dtype=np.float32)
        emb = self.embedder.encode(sentence, convert_to_numpy=True)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def add_sentence(self, sentence: str) -> None:
        """Add a sentence to the FAISS index and internal store."""
        if self.index is None:
            self.texts.append(sentence)
            return
        emb = self._embed(sentence).reshape(1, -1)
        self.index.add(emb)
        self.texts.append(sentence)

    def query_topk(self, sentence: str, k: Optional[int] = None) -> List[Tuple[int, float]]:
        if k is None:
            k = self.k
        if self.index is None or len(self.texts) == 0:
            return []
        emb = self._embed(sentence).reshape(1, -1)
        D, I = self.index.search(emb, k)
        results: List[Tuple[int, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            results.append((int(idx), float(score)))
        return results

    def get_max_contradiction(self, sentence: str) -> float:
        """Return the maximum contradiction probability between `sentence` and top-k past sentences."""
        neighbors = self.query_topk(sentence)
        if not neighbors or self.nli is None:
            return 0.0
        max_contrad = 0.0
        for idx, _ in neighbors:
            past = self.texts[idx]
            preds = self.nli(f"{past} </s> {sentence}")
            if not preds or not preds[0]:
                continue
            for d in preds[0]:
                lab = d.get("label", "").lower()
                if "contradiction" in lab or "contrad" in lab:
                    max_contrad = max(max_contrad, float(d.get("score", 0.0)))
        return float(max_contrad)
