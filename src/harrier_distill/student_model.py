"""Student MiniLM encoder with mean pooling + L2 normalize."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from harrier_distill.performance import attn_implementation


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class StudentEncoder(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        max_length: int = 512,
        perf: dict[str, Any] | None = None,
    ):
        super().__init__()
        perf = perf or {}
        attn = attn_implementation(perf)
        kwargs: dict[str, Any] = {}
        if attn:
            kwargs["attn_implementation"] = attn
        self.encoder = AutoModel.from_pretrained(str(model_path), **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.max_length = max_length
        if perf.get("gradient_checkpointing", True):
            self.encoder.gradient_checkpointing_enable()
            self.encoder.config.use_cache = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # SentenceTransformer-compatible modules.json (mean pooling)
        modules = [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
        ]
        import json

        (output_dir / "modules.json").write_text(json.dumps(modules, indent=2), encoding="utf-8")
        pool_dir = output_dir / "1_Pooling"
        pool_dir.mkdir(exist_ok=True)
        (pool_dir / "config.json").write_text(
            json.dumps(
                {
                    "word_embedding_dimension": self.encoder.config.hidden_size,
                    "pooling_mode_cls_token": False,
                    "pooling_mode_mean_tokens": True,
                    "pooling_mode_max_tokens": False,
                    "pooling_mode_mean_sqrt_len_tokens": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "config_sentence_transformers.json").write_text(
            json.dumps(
                {
                    "prompts": {
                        "sts": "sts: ",
                        "ret": "ret: ",
                        "bitext": "bitext: ",
                    },
                    "default_prompt_name": None,
                    "similarity_fn_name": "cosine",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
