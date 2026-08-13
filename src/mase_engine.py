from typing import Callable, Dict, Optional

from .seg_tracker import SEGTracker
from .adjacent_coherence import AdjacentCoherence
from .topk_graph import TopKGraph


class MASEEngine:
    """Coordinator for the Edge-MASE pipeline."""

    def __init__(self, weights: Optional[Dict[str, float]] = None, threshold: float = 0.5, alarm_callback: Optional[Callable] = None) -> None:
        self.seg = SEGTracker()
        self.adj = AdjacentCoherence()
        self.graph = TopKGraph()
        self.weights = weights or {"w1": 1.0, "w2": 1.0, "w3": 1.0}
        self.threshold = threshold
        self.alarm_callback = alarm_callback

    def reset(self) -> None:
        self.seg = SEGTracker()
        self.adj.reset()
        self.graph.reset()

    def process_sentence(self, sentence: str, token_probs = None) -> Dict[str, float]:
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
                    pass

        return {"seg": seg_val, "h_adjacent": h_adj, "c_long_range": c_long, "risk": float(risk), "alarm": alarm}


class ExtendedMASEEngine:
    """Compatibility wrapper used by evaluation scripts."""

    def __init__(self, w1: float = 1.0, w2: float = 1.0, w4: float = 1.0, threshold: float = 0.5, device: str = "cpu") -> None:

        weights = {"w1": float(w1), "w2": float(w2), "w3": float(w4)}
        # 3. Explicitly pass device=device into the coordinator
        self.engine = MASEEngine(weights=weights, threshold=float(threshold), device=device)

    def reset(self) -> None:
        self.engine.reset()

    def process_new_sentence(self, sentence: str):
        out = self.engine.process_sentence(sentence, token_probs=None)
        try:
            if getattr(self.engine, "graph", None) is not None:
                self.engine.graph.add_sentence(sentence)
        except Exception:
            pass
        return out