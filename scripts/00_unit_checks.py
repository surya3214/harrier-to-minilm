#!/usr/bin/env python3
"""Lightweight offline unit checks (no model downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harrier_distill.losses import DistillLossBundle, disperse_loss, kl_similarity_matrix
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
    s = torch.nn.functional.normalize(torch.randn(8, 384), dim=-1)
    t = torch.nn.functional.normalize(torch.randn(8, 1024), dim=-1)
    # project teacher randomly to make dims unused — losses are cosine-based so ok
    pos = torch.zeros(8, 8, dtype=torch.bool)
    pos[0, 1] = pos[1, 0] = True
    pos[2, 3] = pos[3, 2] = True
    bundle = DistillLossBundle()
    out = bundle(s, t, pos)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(kl_similarity_matrix(s, t))
    assert torch.isfinite(disperse_loss(s))
    print("losses ok", float(out["loss"]))


def main():
    test_schema()
    test_prompts()
    test_losses()
    print("all unit checks passed")


if __name__ == "__main__":
    main()
