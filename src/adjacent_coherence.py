from typing import Optional, List, Dict, Any
import numpy as np
from transformers import pipeline


class AdjacentCoherence:
    """Computes transition entropy between consecutive sentences using an NLI model."""

    def __init__(self, model_name: str = "roberta-large-mnli") -> None:
        try:
            # FIX: Replaced return_all_scores=True with top_k=None
            self.nli = pipeline("text-classification", model=model_name, top_k=None, device=-1)
        except Exception:
            self.nli = None
        self.prev_sentence: Optional[str] = None

    def reset(self) -> None:
        self.prev_sentence = None

    def _query_nli(self, prev: str, cur: str) -> List[Dict[str, Any]]:
        if self.nli is None:
            return []
        try:
            res = self.nli([{"text": prev, "text_pair": cur}])
        except Exception:
            try:
                res = self.nli([(prev, cur)])
            except Exception:
                try:
                    res = self.nli(f"{prev} </s></s> {cur}")
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

    def compute_transition_entropy(self, prev: str, cur: str) -> float:
        score_rows = self._query_nli(prev, cur)
        if not score_rows:
            return 0.0
        scores = np.array([float(d.get("score", 0.0)) for d in score_rows], dtype=float)
        if scores.sum() <= 0:
            return 0.0
        p = scores / scores.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    def on_sentence(self, sentence: str) -> float:
        if self.prev_sentence is None:
            self.prev_sentence = sentence
            return 0.0
        h = self.compute_transition_entropy(self.prev_sentence, sentence)
        self.prev_sentence = sentence
        return float(h)