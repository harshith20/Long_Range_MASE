from typing import Callable, Dict, Optional

from .seg_tracker import SEGTracker
from .adjacent_coherence import AdjacentCoherence
from .topk_graph import TopKGraph


class MASEEngine:
    """Coordinator for the Edge-MASE pipeline.

    Example:
        engine = MASEEngine(weights={"w1":1.0,"w2":1.0,"w3":1.0}, threshold=0.5)
        out = engine.process_sentence(sentence, token_probs=[...])
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, threshold: float = 0.5, alarm_callback: Optional[Callable] = None) -> None:
        self.seg = SEGTracker()
        self.adj = AdjacentCoherence()
        self.graph = TopKGraph()
        self.weights = weights or {"w1": 1.0, "w2": 1.0, "w3": 1.0}
        self.threshold = threshold
        self.alarm_callback = alarm_callback

    def process_sentence(self, sentence: str, token_probs = None) -> Dict[str, float]:
        """Process a single generated sentence through the three signals and compute Risk.

        Args:
            sentence:Generated sentence text.
            token_probs: Optional list of token probabilities for SEG.

        Returns:
            Dict with signal values and aggregate risk.
        """
        seg_val = self.seg.update(token_probs or [])
        h_adj = self.adj.on_sentence(sentence)
        c_long = self.graph.get_max_contradiction(sentence)

        risk = (
            self.weights.get("w1", 1.0) * seg_val
            + self.weights.get("w2", 1.0) * h_adj
            + self.weights.get("w3", 1.0) * c_long
        )

        alarm = False
        if risk >= self.threshold:
            alarm = True
            if self.alarm_callback is not None:
                try:
                    self.alarm_callback(sentence, risk)
                except Exception:
                    # Do not let alarm callback break the pipeline
                    pass

        return {"seg": seg_val, "h_adjacent": h_adj, "c_long_range": c_long, "risk": float(risk), "alarm": alarm}
