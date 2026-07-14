"""Distributed training helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int, torch.device]:
    """Returns rank, world_size, local_rank, device."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, world_size, local_rank, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def all_gather_embeddings(local_emb: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather embeddings; grads flow only through local chunk after cat."""
    if world_size == 1:
        return local_emb
    gathered = [torch.zeros_like(local_emb) for _ in range(world_size)]
    dist.all_gather(gathered, local_emb.contiguous())
    # replace local rank slice with the tensor that has grad
    rank = dist.get_rank()
    gathered[rank] = local_emb
    return torch.cat(gathered, dim=0)


def all_gather_mask(local_mask: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather square masks into a block-diagonal-ish global mask.

    We only gather embeddings globally; positive pairs that cross ranks are ignored
    unless both sides landed on same rank. For cross-rank positives, expand by
    placing local masks on the diagonal blocks.
    """
    if world_size == 1:
        return local_mask
    n = local_mask.size(0)
    gathered = [torch.zeros_like(local_mask) for _ in range(world_size)]
    dist.all_gather(gathered, local_mask.contiguous())
    global_n = n * world_size
    out = local_mask.new_zeros(global_n, global_n)
    for r, block in enumerate(gathered):
        out[r * n : (r + 1) * n, r * n : (r + 1) * n] = block
    return out
