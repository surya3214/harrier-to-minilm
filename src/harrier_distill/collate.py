"""Batch collation for distillation training."""

from __future__ import annotations

from typing import Any

import torch

from harrier_distill.prompts import PromptMap


def _as_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except TypeError:
        return [x]


class DistillCollator:
    def __init__(
        self,
        prompt_map: PromptMap,
        teacher_index,
        max_texts_per_batch: int = 256,
    ):
        self.prompt_map = prompt_map
        self.teacher_index = teacher_index
        self.max_texts_per_batch = max_texts_per_batch

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        student_texts: list[str] = []
        text_ids: list[str] = []
        use_query_flags: list[bool] = []
        positive_pairs: list[tuple[int, int]] = []

        for ex in examples:
            texts = _as_list(ex["texts"])
            roles = _as_list(ex["roles"])
            s_prompts = _as_list(ex["student_prompt_names"])
            tids = _as_list(ex["text_ids"]) if ex.get("text_ids") is not None else None
            if tids is None or any(t is None for t in tids):
                continue
            if any(str(tid) not in self.teacher_index for tid in tids):
                continue

            local_indices = []
            local_roles = []
            for text, role, s_prompt, tid in zip(texts, roles, s_prompts, tids):
                if len(student_texts) >= self.max_texts_per_batch:
                    break
                s_name = s_prompt if s_prompt not in (None, "None", "nan") else None
                if role != "query" and s_name in ("ret", "sts", "bitext"):
                    # passages / second STS sentence: no task prefix
                    if role in ("positive", "negative", "document"):
                        s_name = "document"
                student_texts.append(self.prompt_map.student_text(s_name, str(text)))
                text_ids.append(str(tid))
                use_query = role in ("query", "sentence_a", "sentence_b")
                use_query_flags.append(use_query)
                local_indices.append(len(student_texts) - 1)
                local_roles.append(role)

            if len(local_indices) < 2:
                continue

            role_to_idx: dict[str, list[int]] = {}
            for li, role in zip(local_indices, local_roles):
                role_to_idx.setdefault(role, []).append(li)

            if "query" in role_to_idx and "positive" in role_to_idx:
                for q in role_to_idx["query"]:
                    for p in role_to_idx["positive"]:
                        positive_pairs.append((q, p))
                        positive_pairs.append((p, q))
            if "sentence_a" in role_to_idx and "sentence_b" in role_to_idx:
                for a in role_to_idx["sentence_a"]:
                    for b in role_to_idx["sentence_b"]:
                        positive_pairs.append((a, b))
                        positive_pairs.append((b, a))
            if ex.get("task") == "bitext" and len(local_indices) >= 2:
                a, b = local_indices[0], local_indices[1]
                positive_pairs.append((a, b))
                positive_pairs.append((b, a))
            if ex.get("task") == "sts" and len(local_indices) >= 2:
                a, b = local_indices[0], local_indices[1]
                label = ex.get("label")
                if label is None or (isinstance(label, (int, float)) and float(label) >= 2.5):
                    positive_pairs.append((a, b))
                    positive_pairs.append((b, a))

            if len(student_texts) >= self.max_texts_per_batch:
                break

        n = len(student_texts)
        pos_mask = torch.zeros(n, n, dtype=torch.bool)
        for i, j in positive_pairs:
            if i < n and j < n:
                pos_mask[i, j] = True

        teacher_emb = (
            self.teacher_index.get(text_ids, use_query=use_query_flags)
            if text_ids
            else torch.zeros(0, self.teacher_index.dim)
        )

        return {
            "student_texts": student_texts,
            "text_ids": text_ids,
            "teacher_emb": teacher_emb,
            "positive_mask": pos_mask,
        }
