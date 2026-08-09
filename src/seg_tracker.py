from collections import deque
from typing import Deque, List, Optional

import numpy as np


class SEGTracker:
    """Tracks token probability distributions over a sliding window and computes
    a Semantic Entropy Gradient (SEG) spike metric.

    Typical usage:
        seg = SEGTracker(window=50)
        seg_value = seg.update([0.1, 0.9, ...])
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.token_probs: Deque[List[float]] = deque(maxlen=window)

    def update(self, token_probs: List[float]) -> float:
        """Append a new sequence of token probabilities and return the current SEG.

        Args:
            token_probs: list of token probabilities for the generated sentence

        Returns:
            SEG value (float) representing recent entropy gradient.
        """
        self.token_probs.append(token_probs)
        return float(self.compute_seg())

    @staticmethod
    def compute_entropy(probs: List[float]) -> float:
        p = np.asarray(probs, dtype=float)
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def compute_seg(self) -> float:
        """Compute a simple gradient of the entropy across the most recent items.

        This implementation uses the difference between the last two entropy
        values as a lightweight proxy for a SEG spike.
        """
        if len(self.token_probs) < 2:
            return 0.0
        entropies = [self.compute_entropy(p) for p in self.token_probs]
        return float(entropies[-1] - entropies[-2])
