"""Teacher pooling helpers (Harrier last-token)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def normalize_emb(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)
