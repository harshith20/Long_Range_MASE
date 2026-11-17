"""CPU-only tests for M1 pieces that don't need a model."""

from __future__ import annotations

import numpy as np
import pytest

from hallu_hierarchy.data.normalize import normalize_answer
from hallu_hierarchy.data.prepare import select_questions
from hallu_hierarchy.data.splits import assign_subject_splits, check_no_leakage
from hallu_hierarchy.generation.checkpoint import ShardStore
from hallu_hierarchy.generation.prompts import subject_char_span, tokens_in_span


def test_normalize():
    assert normalize_answer("The  Beatles!") == "beatles"
    assert normalize_answer("São Paulo") == "sao paulo"
    assert normalize_answer("New-York") == "new york"
    assert normalize_answer("an apple, a day") == "apple day"


def test_subject_splits_no_leakage():
    items = [{"qid": i, "subject_id": f"s{i % 7}"} for i in range(50)]
    out = assign_subject_splits(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    check_no_leakage(out)
    assert {o["split"] for o in out} == {"train", "val", "test"}
    assert out == assign_subject_splits(items, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)


def test_leakage_detected():
    with pytest.raises(AssertionError):
        check_no_leakage([{"subject_id": "a", "split": "train"}, {"subject_id": "a", "split": "test"}])


def test_select_questions_deterministic():
    items = [{"qid": i} for i in range(100)]
    a, b = select_questions(items, 10, 0), select_questions(items, 10, 0)
    assert a == b and len(a) == 10
    assert select_questions(items, None, 0) == items


def test_subject_token_span():
    question = "What is George Rankin's occupation?"
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    span = subject_char_span(prompt, question, "george rankin")
    assert prompt[span[0]:span[1]] == "George Rankin"
    # fake offsets: tokens "What"," is"," George"," Rankin","'s"
    base = prompt.find(question)
    offsets = [(0, 0), (base, base + 4), (base + 4, base + 7), (base + 7, base + 14),
               (base + 14, base + 21), (base + 21, base + 23)]
    assert tokens_in_span(offsets, span) == [3, 4]
    assert subject_char_span(prompt, question, "Einstein") is None


def test_shard_store_resume(tmp_path):
    store = ShardStore(tmp_path / "a", chunk_size=4, fp="abc")
    assert store.missing_chunks(10) == [0, 1, 2]
    assert list(store.chunk_range(2, 10)) == [8, 9]
    store.save_chunk(0, [{"qid": i} for i in range(4)], {"x": np.ones((4, 3), np.float16)})
    assert store.missing_chunks(10) == [1, 2]

    recs, arrs = store.load_chunk(0)
    assert len(recs) == 4 and arrs["x"].dtype == np.float16

    # a new store elsewhere can import the finished shard
    other = ShardStore(tmp_path / "b", chunk_size=4, fp="abc")
    assert other.import_from([tmp_path / "a"]) == 1
    assert other.missing_chunks(10) == [1, 2]

    with pytest.raises(RuntimeError):
        ShardStore(tmp_path / "a", chunk_size=4, fp="different")
