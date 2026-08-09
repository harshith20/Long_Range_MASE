from typing import Optional

import numpy as np

from transformers import pipeline


class AdjacentCoherence:
    """Computes transition entropy between consecutive sentences using an NLI model.

    The class uses a classification `pipeline` (roberta-large-mnli) that returns
    scores for labels such as ENTAILMENT / NEUTRAL / CONTRADICTION. We treat
    those scores as a discrete distribution and compute Shannon entropy.
    """

    def __init__(self, model_name: str = "roberta-large-mnli") -> None:
        try:
            self.nli = pipeline("text-classification", model=model_name, return_all_scores=True)
        except Exception:
            self.nli = None
        self.prev_sentence: Optional[str] = None

    def compute_transition_entropy(self, prev: str, cur: str) -> float:
        if self.nli is None:
            return 0.0
        # Compose a short premise-hypothesis pair and query the NLI model
        preds = self.nli(f"{prev} </s> {cur}")
        if not preds or not preds[0]:
            return 0.0
        scores = np.array([d["score"] for d in preds[0]], dtype=float)
        if scores.sum() <= 0:
            return 0.0
        p = scores / scores.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    def on_sentence(self, sentence: str) -> float:
        """Process a new sentence and return H_adjacent relative to the previous sentence.

        The first call returns 0.0 (no previous sentence).
        """
        if self.prev_sentence is None:
            self.prev_sentence = sentence
            return 0.0
        h = self.compute_transition_entropy(self.prev_sentence, sentence)
        self.prev_sentence = sentence
        return float(h)
