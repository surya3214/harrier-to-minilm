#!/usr/bin/env python3
"""Multi-GPU student distillation with KL + focal InfoNCE + disperse."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harrier_distill.batching import apply_valid_mask_to_pos, pad_embeddings
from harrier_distill.collate import DistillCollator
from harrier_distill.config import load_yaml, project_root, resolve_path
from harrier_distill.data import ExampleShardDataset, TeacherEmbeddingIndex
from harrier_distill.distributed import (
    all_gather_embeddings,
    all_gather_mask,
    barrier,
    cleanup,
    init_distributed,
    is_main,
)
from harrier_distill.losses import DistillLossBundle
from harrier_distill.performance import (
    apply_torch_performance,
    build_adamw,
    dataloader_kwargs,
    move_batch,
)
from harrier_distill.prompts import PromptMap
from harrier_distill.student_model import StudentEncoder


def cosine_lr(step: int, total: int, base_lr: float, warmup_ratio: float) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/en_ko_sts_retrieval.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    perf = cfg.get("performance", {})
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    apply_torch_performance(perf)

    rank, world_size, local_rank, device = init_distributed()

    shards_root = resolve_path(cfg, "shards_root", root)
    emb_root = resolve_path(cfg, "embeddings_root", root)
    outputs_root = resolve_path(cfg, "outputs_root", root)
    if is_main(rank):
        outputs_root.mkdir(parents=True, exist_ok=True)

    student_path = root / cfg["models"]["student_local"]
    if not student_path.exists():
        student_path = Path(cfg["models"]["student"])

    prompt_map = PromptMap.from_config(cfg)
    teacher_index = TeacherEmbeddingIndex(emb_root)
    dataset = ExampleShardDataset(shards_root, split="train")
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    max_texts = int(train_cfg.get("target_embeddings_n", 256))
    collator = DistillCollator(prompt_map, teacher_index, max_texts_per_batch=max_texts)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        drop_last=True,
        **dataloader_kwargs(perf, int(train_cfg.get("num_workers", 4))),
    )

    student = StudentEncoder(
        student_path, max_length=int(cfg.get("max_length", 512)), perf=perf
    )
    student.to(device)
    if world_size > 1:
        student = DDP(student, device_ids=[local_rank] if device.type == "cuda" else None)

    raw_student = student.module if isinstance(student, DDP) else student
    opt = build_adamw(
        raw_student.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        fused=bool(perf.get("fused_adamw", True)),
    )
    loss_fn = DistillLossBundle(
        lambda_kl=float(loss_cfg.get("lambda_kl", 1.0)),
        lambda_focal=float(loss_cfg.get("lambda_focal", 0.5)),
        lambda_disp=float(loss_cfg.get("lambda_disp", 0.05)),
        temperature=float(loss_cfg.get("temperature", 0.05)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        disperse_t=float(loss_cfg.get("disperse_t", 2.0)),
    )

    max_steps = int(args.max_steps or train_cfg.get("max_steps", 50000))
    base_lr = float(train_cfg.get("lr", 2e-5))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 2000))
    use_bf16 = bool(perf.get("bf16", True)) and device.type == "cuda"
    non_blocking = bool(perf.get("non_blocking", True))

    step = 0
    student.train()
    if is_main(rank):
        print(
            f"Training student from {student_path} | world_size={world_size} "
            f"| FA2={perf.get('flash_attention_2', False)} | max_steps={max_steps}"
        )

    t0 = time.time()
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            if step >= max_steps:
                break
            if not batch["student_texts"]:
                continue

            tok = raw_student.tokenize(batch["student_texts"])
            tok = move_batch(tok, device, non_blocking)
            teacher_emb = batch["teacher_emb"].to(device, non_blocking=non_blocking)
            pos_mask = batch["positive_mask"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                local_emb = student(tok["input_ids"], tok["attention_mask"])

            # Pad to fixed N so DDP all-gather shapes match across ranks
            local_emb_f = local_emb.float()
            teacher_emb_f = teacher_emb.float()
            if local_emb_f.size(0) != teacher_emb_f.size(0):
                continue
            n0 = local_emb_f.size(0)
            local_emb_f, pos_mask, valid = pad_embeddings(local_emb_f, pos_mask, max_texts)
            teacher_emb_f, _, _ = pad_embeddings(
                teacher_emb_f,
                teacher_emb_f.new_zeros(n0, n0).bool(),
                max_texts,
            )
            pos_mask = apply_valid_mask_to_pos(pos_mask, valid)

            global_student = all_gather_embeddings(local_emb_f, world_size)
            global_teacher = all_gather_embeddings(teacher_emb_f, world_size)
            global_mask = all_gather_mask(pos_mask, world_size)
            global_valid = (
                all_gather_embeddings(valid.float().unsqueeze(-1), world_size).squeeze(-1) > 0.5
            )
            global_mask = apply_valid_mask_to_pos(global_mask, global_valid)

            out = loss_fn(global_student, global_teacher, global_mask)
            loss = out["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_student.parameters(), grad_clip)
            lr = cosine_lr(step, max_steps, base_lr, warmup_ratio)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.step()

            if is_main(rank) and step % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"step={step} loss={loss.item():.4f} kl={out['loss_kl'].item():.4f} "
                    f"focal={out['loss_focal'].item():.4f} disp={out['loss_disp'].item():.4f} "
                    f"lr={lr:.2e} n={global_student.size(0)} elapsed={elapsed:.1f}s"
                )

            if is_main(rank) and step > 0 and step % save_every == 0:
                ckpt = outputs_root / f"checkpoint-{step}"
                raw_student.save_pretrained(ckpt)
                print(f"Saved {ckpt}")

            step += 1

    barrier()
    if is_main(rank):
        final = outputs_root / "student_final"
        raw_student.save_pretrained(final)
        print(f"Training complete. Final model -> {final}")
    cleanup()


if __name__ == "__main__":
    main()
