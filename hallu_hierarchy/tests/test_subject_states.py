"""CPU tests for the subject-state stage helpers (no model download)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hallu_hierarchy.generation.subject_states import (
    compare_with_generation, relative_diff, subject_last_index,
)
from hallu_hierarchy.run import STAGES, parse_stages


def test_stage_registered():
    assert "subject_states" in STAGES
    assert parse_stages(["subject_states"]) == ["subject_states"]


def test_subject_last_index():
    assert subject_last_index([5, 6, 7]) == 7
    assert subject_last_index([9, 3]) == 9
    assert subject_last_index([]) is None


def test_compare_with_generation():
    gen = {"q1": {"prompt_len": 40, "subject_token_idx": [20, 21]},
           "q2": {"prompt_len": 35, "subject_token_idx": []}}
    ok = [{"qid": "q1", "prompt_len": 40, "subject_token_idx": [20, 21]},
          {"qid": "q2", "prompt_len": 35, "subject_token_idx": []}]
    assert compare_with_generation(ok, gen) == []
    bad = [{"qid": "q1", "prompt_len": 41, "subject_token_idx": [20, 21]},
           {"qid": "q3", "prompt_len": 30, "subject_token_idx": [1]}]
    msgs = compare_with_generation(bad, gen)
    assert len(msgs) == 2 and "q1" in msgs[0] and "not in generations" in msgs[1]


def test_relative_diff():
    a = np.array([[1000.0, 2.0]], dtype=np.float16)
    assert relative_diff(a, a) == 0.0
    b = a.copy()
    b[0, 1] = 12.0                       # off by 10 on a scale of 1000
    assert abs(relative_diff(b, a) - 0.01) < 1e-6


def test_prompt_states_with_fake_model():
    torch = pytest.importorskip("torch")
    from hallu_hierarchy.generation.subject_states import prompt_states

    L, d = 5, 4

    class Fake:
        device = "cpu"

        def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
            T = input_ids.shape[1]
            # hidden state value = layer * 100 + position, so we can see what was picked
            hs = tuple(torch.full((1, T, d), 0.0) + layer * 100 + torch.arange(T)[None, :, None]
                       for layer in range(L))
            return SimpleNamespace(hidden_states=hs)

    subj, tbg = prompt_states(Fake(), [11, 12, 13, 14, 15, 16], positions=[3, 5], layers=[0, 2, 4])
    assert subj.shape == (3, d) and subj.dtype == np.float16
    assert subj[:, 0].tolist() == [3.0, 203.0, 403.0]     # layers 0,2,4 at position 3
    assert tbg[:, 0].tolist() == [5.0, 205.0, 405.0]      # last prompt token
