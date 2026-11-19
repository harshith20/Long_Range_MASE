"""Bidirectional entailment with an MNLI model (default microsoft/deberta-large-mnli)."""

from __future__ import annotations

from typing import Any


class NLI:
    def __init__(self, model_id: str, device: str = "cuda", batch_size: int = 32) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # Weights stay fp32: DeBERTa mixes fp32 tensors into its attention, so a model cast to fp16
        # crashes ("expected scalar type Float but found Half", kernel v9). Speed comes from autocast.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.use_autocast = self.device == "cuda"
        self.batch_size = batch_size
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"{model_id} has no 'entailment' label: {self.model.config.id2label}")
        self.entail_id = labels["entailment"]
        self.id2label = dict(self.model.config.id2label)
        self.n_pairs = 0
        self.switched_to_fp32 = False
        self.fallback_reason: str | None = None

    def _forward(self, enc: Any, autocast: bool) -> Any:
        torch = self.torch
        with torch.no_grad():
            if autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(**enc).logits.float()
            return self.model(**enc).logits.float()

    def _logits(self, enc: Any) -> Any:
        """Mixed precision on the GPU; on any error or non-finite logit, fall back to plain fp32 for good."""
        torch = self.torch
        if self.use_autocast:
            try:
                logits = self._forward(enc, autocast=True)
                if torch.isfinite(logits).all():
                    return logits
                reason = "non-finite logits"
            except RuntimeError as e:
                reason = f"{type(e).__name__}: {str(e)[:120]}"
            self.use_autocast = False
            self.switched_to_fp32 = True
            self.fallback_reason = reason
        logits = self._forward(enc, autocast=False)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("NLI produced non-finite logits in fp32")
        return logits

    def sanity(self) -> dict[str, bool]:
        """Three fixed pairs with obvious answers; all must come out as expected."""
        q = "What is the capital of France?"
        got = self.entails([(f"{q} Paris", f"{q} Paris"), (f"{q} Paris", f"{q} Lyon"),
                            (f"{q} Paris", f"{q} The city of Paris")])
        return {"same_entails": got[0], "different_not": not got[1], "paraphrase_entails": got[2]}

    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        """For each (premise, hypothesis): is the predicted label 'entailment'?"""
        out: list[bool] = []
        torch = self.torch
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], return_tensors="pt",
                           padding=True, truncation=True, max_length=256).to(self.device)
            pred = self._logits(enc).argmax(-1).tolist()
            out += [p == self.entail_id for p in pred]
        self.n_pairs += len(pairs)
        return out


def directed_pairs(answers: list[str], context: str | None) -> list[tuple[int, int, str, str]]:
    """All ordered pairs (i, j), i != j, as NLI inputs. With a context (the question), each answer is
    read as "question answer", as in semantic entropy."""
    def text(a: str) -> str:
        return f"{context} {a}" if context else a

    return [(i, j, text(answers[i]), text(answers[j]))
            for i in range(len(answers)) for j in range(len(answers)) if i != j]


def entail_lookup(pairs: list[tuple[int, int, str, str]], results: list[bool]) -> Any:
    table = {(i, j): r for (i, j, _, _), r in zip(pairs, results)}
    return lambda i, j: table[(i, j)]
