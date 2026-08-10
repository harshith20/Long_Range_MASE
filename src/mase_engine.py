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


class ExtendedMASEEngine:
    """Compatibility wrapper used by evaluation scripts.

    This class exposes a simpler initializer accepting explicit `w1`, `w2`,
    and `w4` (long-range/top-k) weights and forces transformer pipelines
    and sentence-transformer embedder to CPU where possible.
    """

    def __init__(self, w1: float = 1.0, w2: float = 1.0, w4: float = 1.0, threshold: float = 0.5, device: str = "cpu") -> None:
        # Map w4 -> w3 used inside the original MASEEngine
        weights = {"w1": float(w1), "w2": float(w2), "w3": float(w4)}
        self.engine = MASEEngine(weights=weights, threshold=float(threshold))

        # Attempt to reconfigure internal NLI/embedder components to run on CPU.
        try:
            from transformers import pipeline
        except Exception:
            pipeline = None

        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            SentenceTransformer = None

        # Force CPU for transformers pipelines by using device=-1 if pipeline available
        if pipeline is not None:
            try:
                if getattr(self.engine, "adj", None) is not None:
                    # Recreate or replace the AdjacentCoherence NLI pipeline on CPU
                    try:
                        self.engine.adj.nli = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True, device=-1)
                    except Exception:
                        # leave whatever was created originally
                        pass

                if getattr(self.engine, "graph", None) is not None:
                    try:
                        self.engine.graph.nli = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True, device=-1)
                    except Exception:
                        pass
            except Exception:
                pass

        # Force SentenceTransformer to CPU if available and rebuild index
        if SentenceTransformer is not None and getattr(self.engine, "graph", None) is not None:
            try:
                # Recreate embedder on CPU and rebuild index structure
                model_name = getattr(self.engine.graph, "embedder", None)
                # model_name may be an instance; fallback to default name
                model_name_str = "all-MiniLM-L6-v2"
                try:
                    new_embedder = SentenceTransformer(model_name_str, device="cpu")
                    self.engine.graph.embedder = new_embedder
                    self.engine.graph.dim = new_embedder.get_sentence_embedding_dimension()
                    try:
                        import faiss

                        self.engine.graph.index = faiss.IndexFlatIP(self.engine.graph.dim)
                    except Exception:
                        # leave index as-is if faiss not available
                        pass
                except Exception:
                    pass
            except Exception:
                pass

    def process_new_sentence(self, sentence: str):
        """Process a single sentence through the underlying engine and
        append it to the long-range graph store afterwards.
        Returns the same dict produced by `MASEEngine.process_sentence`.
        """
        out = self.engine.process_sentence(sentence, token_probs=None)
        # Add sentence to history for future long-range checks
        try:
            if getattr(self.engine, "graph", None) is not None:
                try:
                    self.engine.graph.add_sentence(sentence)
                except Exception:
                    # Non-fatal: ignore failures to add to graph
                    pass
        except Exception:
            pass
        return out

