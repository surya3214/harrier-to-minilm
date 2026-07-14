#!/usr/bin/env python3
"""Multi-GPU Harrier teacher embedding over texts.parquet (air-gap GPU host)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harrier_distill.config import load_yaml, project_root, resolve_path
from harrier_distill.data import ExampleShardDataset, load_texts_dataframe
from harrier_distill.distributed import barrier, cleanup, init_distributed, is_main
from harrier_distill.performance import (
    apply_torch_performance,
    attn_implementation,
    dataloader_kwargs,
    move_batch,
)
from harrier_distill.prompts import PromptMap
from harrier_distill.teacher_pool import last_token_pool, normalize_emb


class TextsDataset(Dataset):
    def __init__(self, ids, texts):
        self.ids = ids
        self.texts = texts

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {"text_id": self.ids[idx], "text": self.texts[idx], "row": idx}


def collate_texts(batch):
    return {
        "text_id": [b["text_id"] for b in batch],
        "text": [b["text"] for b in batch],
        "row": [b["row"] for b in batch],
    }


def encode_batches(
    model,
    tokenizer,
    loader,
    mm,
    device,
    max_length: int,
    use_bf16: bool,
    non_blocking: bool,
    rank: int,
):
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, disable=not is_main(rank), desc=f"embed[rank{rank}]"):
            enc = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = move_batch(enc, device, non_blocking)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(**enc)
                emb = normalize_emb(
                    last_token_pool(out.last_hidden_state, enc["attention_mask"]).float()
                )
            emb_np = emb.detach().cpu().numpy().astype(np.float16)
            for row_i, vec in zip(batch["row"], emb_np):
                mm[int(row_i)] = vec
            mm.flush()


def collect_query_prompted_rows(
    shards_root: Path, text_id_to_row: dict[str, int], prompt_map: PromptMap
) -> list[tuple[int, str]]:
    """Return (row, prompted_text) for texts that need Harrier query instructions."""
    out: dict[int, str] = {}
    try:
        ex_ds = ExampleShardDataset(shards_root, split="train")
    except FileNotFoundError:
        return []
    for i in range(len(ex_ds)):
        ex = ex_ds[i]
        texts = ex["texts"] if isinstance(ex["texts"], list) else list(ex["texts"])
        roles = ex["roles"] if isinstance(ex["roles"], list) else list(ex["roles"])
        t_prompts = (
            ex["teacher_prompt_names"]
            if isinstance(ex["teacher_prompt_names"], list)
            else list(ex["teacher_prompt_names"])
        )
        tids = ex["text_ids"] if isinstance(ex["text_ids"], list) else list(ex["text_ids"])
        for text, role, tp, tid in zip(texts, roles, t_prompts, tids):
            tp_s = None if tp is None else str(tp)
            if tp_s in (None, "document", "None", "nan"):
                continue
            if role not in ("query", "sentence_a", "sentence_b"):
                continue
            tid_s = str(tid)
            if tid_s not in text_id_to_row:
                continue
            row = text_id_to_row[tid_s]
            out[row] = prompt_map.teacher_text(tp_s, str(text))
    return sorted(out.items(), key=lambda x: x[0])


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/en_ko_sts_retrieval.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    perf = cfg.get("performance", {})
    apply_torch_performance(perf)

    rank, world_size, local_rank, device = init_distributed()
    shards_root = resolve_path(cfg, "shards_root", root)
    emb_root = resolve_path(cfg, "embeddings_root", root)
    emb_root.mkdir(parents=True, exist_ok=True)

    teacher_path = root / cfg["models"]["teacher_local"]
    if not teacher_path.exists():
        teacher_path = Path(cfg["models"]["teacher"])

    attn = attn_implementation(perf)
    model_kwargs = {"trust_remote_code": True}
    if attn:
        model_kwargs["attn_implementation"] = attn

    if is_main(rank):
        print(f"Loading teacher from {teacher_path} (flash_attention_2={bool(attn)})")

    tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(teacher_path), **model_kwargs)
    model.eval()
    model.to(device)

    df = load_texts_dataframe(shards_root).reset_index(drop=True)
    n = len(df)
    dim = int(cfg.get("embedding_dim_teacher", 1024))
    prompt_map = PromptMap.from_config(cfg)
    max_length = int(cfg.get("max_length", 512))
    use_bf16 = bool(perf.get("bf16", True)) and device.type == "cuda"
    non_blocking = bool(perf.get("non_blocking", True))
    batch_size = int(cfg.get("embed", {}).get("batch_size", 64))

    # Pass 1: document / raw embeddings (no instruct prefix)
    ds = TextsDataset(df["text_id"].tolist(), df["text"].astype(str).tolist())
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_texts,
        **dataloader_kwargs(perf, int(cfg.get("train", {}).get("num_workers", 2))),
    )

    memmap_path = emb_root / "teacher_emb.fp16.memmap"
    if is_main(rank):
        mm = np.memmap(memmap_path, dtype=np.float16, mode="w+", shape=(n, dim))
        mm.flush()
        del mm
    barrier()

    mm = np.memmap(memmap_path, dtype=np.float16, mode="r+", shape=(n, dim))
    encode_batches(
        model, tokenizer, loader, mm, device, max_length, use_bf16, non_blocking, rank
    )
    barrier()

    text_id_to_row = {tid: i for i, tid in enumerate(df["text_id"].tolist())}
    if is_main(rank):
        (emb_root / "text_id_to_row.json").write_text(
            json.dumps(text_id_to_row), encoding="utf-8"
        )

    # Pass 2: overlay Harrier-instruct embeddings for query-like roles into a twin memmap
    # Training uses teacher_emb_query for query/sentence roles and teacher_emb for passages.
    query_mm_path = emb_root / "teacher_emb_query.fp16.memmap"
    if is_main(rank):
        qmm = np.memmap(query_mm_path, dtype=np.float16, mode="w+", shape=(n, dim))
        qmm[:] = np.memmap(memmap_path, dtype=np.float16, mode="r", shape=(n, dim))[:]
        qmm.flush()
        del qmm
    barrier()
    qmm = np.memmap(query_mm_path, dtype=np.float16, mode="r+", shape=(n, dim))

    items = collect_query_prompted_rows(shards_root, text_id_to_row, prompt_map)
    items = items[rank::world_size]
    for start in tqdm(
        range(0, len(items), batch_size),
        disable=not is_main(rank),
        desc=f"query-embed[rank{rank}]",
    ):
        chunk = items[start : start + batch_size]
        if not chunk:
            continue
        rows = [r for r, _ in chunk]
        texts = [t for _, t in chunk]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = move_batch(enc, device, non_blocking)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**enc)
            emb = normalize_emb(
                last_token_pool(out.last_hidden_state, enc["attention_mask"]).float()
            )
        emb_np = emb.detach().cpu().numpy().astype(np.float16)
        for row_i, vec in zip(rows, emb_np):
            qmm[int(row_i)] = vec
        qmm.flush()

    barrier()
    if is_main(rank):
        meta = {
            "n": n,
            "dim": dim,
            "dtype": "float16",
            "doc_memmap": "teacher_emb.fp16.memmap",
            "query_memmap": "teacher_emb_query.fp16.memmap",
            "flash_attention_2": bool(attn),
        }
        (emb_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote doc+query embeddings for {n} texts under {emb_root}")

    cleanup()


if __name__ == "__main__":
    main()
