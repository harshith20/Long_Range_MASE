from typing import List, Optional, Tuple, Dict, Any
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
    against top-k nearest past sentences to surface contradiction scores.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", nli_model: str = "roberta-large-mnli", k: int = 2) -> None:
        self.k = k
        self.embedder = SentenceTransformer(model_name, device="cpu") if SentenceTransformer is not None else None
        if self.embedder is not None:
            if hasattr(self.embedder, "get_embedding_dimension"):
                self.dim = self.embedder.get_embedding_dimension()
            else:
                self.dim = self.embedder.get_sentence_embedding_dimension()
        else:
            self.dim = 384

        self.index = faiss.IndexFlatIP(self.dim) if (faiss is not None and self.dim > 0) else None
        self.texts: List[str] = []
        self.nli = None
        try:
            self.nli = pipeline("text-classification", model=nli_model, return_all_scores=True, device=-1)
        except Exception:
            self.nli = None

    def reset(self) -> None:
        self.texts = []
        if faiss is not None and self.dim > 0:
            self.index = faiss.IndexFlatIP(self.dim)

    def _embed(self, sentence: str) -> np.ndarray:
        if self.embedder is None:
            return np.zeros(self.dim, dtype=np.float32)
        emb = self.embedder.encode(sentence, convert_to_numpy=True)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def add_sentence(self, sentence: str) -> None:
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
        D, I = self.index.search(emb, min(k, len(self.texts)))
        results: List[Tuple[int, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            results.append((int(idx), float(score)))
        return results

    def _query_nli(self, past: str, cur: str) -> List[Dict[str, Any]]:
        if self.nli is None:
            return []
        try:
            res = self.nli([{"text": past, "text_pair": cur}])
        except Exception:
            try:
                res = self.nli([(past, cur)])
            except Exception:
                try:
                    res = self.nli(f"{past} </s></s> {cur}")
                except Exception:
                    return []

        if not res:
            return []
        if isinstance(res, list) and len(res) > 0:
            if isinstance(res[0], list):
                return res[0]
            elif isinstance(res[0], dict):
                return res
        return []

    def get_max_contradiction(self, sentence: str) -> float:
        neighbors = self.query_topk(sentence)
        if not neighbors or self.nli is None:
            return 0.0
        max_contrad = 0.0
        for idx, _ in neighbors:
            past = self.texts[idx]
            score_rows = self._query_nli(past, sentence)
            if not score_rows:
                continue
            for d in score_rows:
                lab = str(d.get("label", "")).lower()
                if "contradiction" in lab or "contrad" in lab or lab == "label_0":
                    max_contrad = max(max_contrad, float(d.get("score", 0.0)))
        return float(max_contrad)