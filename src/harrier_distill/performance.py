"""GPU performance helpers (FA2 off by default; other accelerations on)."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def apply_torch_performance(perf: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        return
    if perf.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if perf.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True


def attn_implementation(perf: dict[str, Any]) -> str | None:
    """Return transformers attn_implementation or None for default SDPA/eager."""
    if perf.get("flash_attention_2", False):
        return "flash_attention_2"
    return None


def build_adamw(params, lr: float, weight_decay: float, fused: bool = True):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, fused=True, **kwargs)
        except (TypeError, RuntimeError) as e:
            logger.warning("fused AdamW unavailable (%s); falling back", e)
    return torch.optim.AdamW(params, **kwargs)


def dataloader_kwargs(perf: dict[str, Any], num_workers: int) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "pin_memory": bool(perf.get("pin_memory", True)) and torch.cuda.is_available(),
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kw["persistent_workers"] = bool(perf.get("persistent_workers", True))
        kw["prefetch_factor"] = 2
    return kw


def move_batch(batch: dict[str, torch.Tensor], device: torch.device, non_blocking: bool) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out
