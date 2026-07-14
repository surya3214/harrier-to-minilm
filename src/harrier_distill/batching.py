"""Pad variable-length embedding batches for DDP all-gather."""

from __future__ import annotations

import torch


def pad_embeddings(
    emb: torch.Tensor,
    mask: torch.Tensor,
    target_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad to target_n along batch dim.

    Returns padded_emb, padded_pos_mask, valid_mask [target_n] (True = real row).
    """
    n, d = emb.shape
    device = emb.device
    valid = torch.zeros(target_n, dtype=torch.bool, device=device)
    if n >= target_n:
        return emb[:target_n], mask[:target_n, :target_n], torch.ones(target_n, dtype=torch.bool, device=device)

    out = emb.new_zeros(target_n, d)
    out[:n] = emb
    valid[:n] = True
    pos = mask.new_zeros(target_n, target_n)
    pos[:n, :n] = mask
    return out, pos, valid


def apply_valid_mask_to_pos(pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Zero out pairs involving padded rows."""
    v = valid.unsqueeze(0) & valid.unsqueeze(1)
    return pos & v
