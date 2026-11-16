"""Prompt construction (model chat template) and subject-token location."""

from __future__ import annotations

from typing import Any


def build_prompt(tokenizer: Any, question: str, system_prompt: str | None) -> str:
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def subject_char_span(prompt: str, question: str, subject: str | None) -> tuple[int, int] | None:
    """Character span of the subject inside the question part of the prompt (case-insensitive)."""
    if not subject:
        return None
    q_start = prompt.rfind(question)
    if q_start < 0:
        return None
    s = question.lower().find(subject.lower())
    if s < 0:
        return None
    return q_start + s, q_start + s + len(subject)


def tokens_in_span(offsets: list[tuple[int, int]], span: tuple[int, int] | None) -> list[int]:
    if span is None:
        return []
    a, b = span
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]


def encode_prompt(
    tokenizer: Any, question: str, subject: str | None, system_prompt: str | None
) -> tuple[list[int], list[int]]:
    """Return (prompt token ids, indices of prompt tokens covering the subject)."""
    prompt = build_prompt(tokenizer, question, system_prompt)
    # The chat template already contains any special tokens, so don't add them again.
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    span = subject_char_span(prompt, question, subject)
    return list(enc["input_ids"]), tokens_in_span(list(enc["offset_mapping"]), span)
