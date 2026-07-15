#!/usr/bin/env python3
"""Lightweight offline unit checks (no model downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harrier_distill.collate import DistillCollator, _should_use_query_teacher
from harrier_distill.losses import (
    DistillLossBundle,
    disperse_loss,
    kl_similarity_matrix,
    sts_pair_cosine_mse,
)
from harrier_distill.prompts import PromptMap
from harrier_distill.schema import make_example, validate_example


def test_schema():
    ex = make_example(
        "id1",
        "sts",
        "en",
        ["a", "b"],
        ["sentence_a", "sentence_b"],
        ["sts_query", "document"],
        ["sts", "document"],
        label=3.0,
    )
    validate_example(ex)
    print("schema ok")


def test_prompts():
    pm = PromptMap.from_config(
        {
            "prompts": {
                "teacher": {"sts_query": "Instruct: STS\nQuery: "},
                "student": {"sts": "sts: "},
            }
        }
    )
    assert pm.teacher_text("sts_query", "hello").startswith("Instruct:")
    assert pm.student_text("sts", "hello") == "sts: hello"
    assert pm.teacher_text("document", "hello") == "hello"
    print("prompts ok")


def test_losses():
    torch.manual_seed(0)
    s = F.normalize(torch.randn(8, 384), dim=-1)
    t = F.normalize(torch.randn(8, 1024), dim=-1)
    pos = torch.zeros(8, 8, dtype=torch.bool)
    pos[0, 1] = pos[1, 0] = True
    pos[2, 3] = pos[3, 2] = True
    valid = torch.ones(8, dtype=torch.bool)
    pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    bundle = DistillLossBundle()
    out = bundle(s, t, pos, valid=valid, sts_pairs=pairs)
    assert torch.isfinite(out["loss"])
    assert "loss_sts" in out
    assert torch.isfinite(kl_similarity_matrix(s, t, valid=valid))
    assert torch.isfinite(disperse_loss(s, valid=valid))
    print("losses ok", float(out["loss"]), "sts", float(out["loss_sts"]))


def test_pad_masking():
    """Padded zero rows must not dominate KL/disperse when masked."""
    torch.manual_seed(1)
    n_real, n_pad, d = 4, 12, 32
    s_real = F.normalize(torch.randn(n_real, d), dim=-1)
    t_real = F.normalize(torch.randn(n_real, d), dim=-1)
    s = torch.zeros(n_real + n_pad, d)
    t = torch.zeros(n_real + n_pad, d)
    s[:n_real] = s_real
    t[:n_real] = t_real
    valid = torch.zeros(n_real + n_pad, dtype=torch.bool)
    valid[:n_real] = True

    kl_masked = kl_similarity_matrix(s, t, temperature=0.07, valid=valid)
    disp_masked = disperse_loss(s, t=2.0, valid=valid)
    kl_real = kl_similarity_matrix(s_real, t_real, temperature=0.07)
    disp_real = disperse_loss(s_real, t=2.0)

    assert torch.isfinite(kl_masked) and torch.isfinite(disp_masked)
    assert torch.allclose(kl_masked, kl_real, rtol=1e-3, atol=1e-3)
    assert torch.allclose(disp_masked, disp_real, rtol=1e-3, atol=1e-3)
    print("pad masking ok", float(kl_masked), float(disp_masked))


def test_sts_mse_zero_when_matched():
    torch.manual_seed(2)
    s = F.normalize(torch.randn(6, 16), dim=-1)
    # Same geometry in teacher space (reuse student vectors as teacher)
    t = s.clone()
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    loss = sts_pair_cosine_mse(s, t, pairs)
    assert float(loss) < 1e-6
    print("sts mse ok", float(loss))


def test_collate_sts_query_flags():
    class FakeIndex:
        dim = 8

        def __contains__(self, tid):
            return True

        def get(self, text_ids, use_query=None):
            assert use_query is not None
            assert use_query == [True, False], use_query
            return torch.zeros(len(text_ids), self.dim)

    assert _should_use_query_teacher("sts", "sentence_a", 0) is True
    assert _should_use_query_teacher("sts", "sentence_b", 1) is False
    assert _should_use_query_teacher("retrieval", "positive", 1) is False

    pm = PromptMap.from_config({})
    collator = DistillCollator(pm, FakeIndex(), max_texts_per_batch=16)
    ex = make_example(
        "sts-0",
        "sts",
        "en",
        ["hello world", "hi earth"],
        ["sentence_a", "sentence_b"],
        ["sts_query", "document"],
        ["sts", "document"],
        label=4.0,
        text_ids=["en_a", "en_b"],
    )
    batch = collator([ex])
    assert batch["use_query_flags"] == [True, False]
    assert batch["sts_pairs"].shape == (1, 2)
    assert tuple(batch["sts_pairs"][0].tolist()) == (0, 1)
    print("collate sts roles ok")


def main():
    test_schema()
    test_prompts()
    test_losses()
    test_pad_masking()
    test_sts_mse_zero_when_matched()
    test_collate_sts_query_flags()
    print("all unit checks passed")


if __name__ == "__main__":
    main()
