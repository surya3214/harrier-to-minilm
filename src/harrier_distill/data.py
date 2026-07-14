"""Shard datasets and teacher embedding index access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class TeacherEmbeddingIndex:
    """Memory-mapped teacher embeddings keyed by text_id.

    Uses query-prompted memmap for query-like roles when available, otherwise doc memmap.
    """

    def __init__(self, embeddings_root: str | Path):
        self.root = Path(embeddings_root)
        meta_path = self.root / "meta.json"
        index_path = self.root / "text_id_to_row.json"
        memmap_path = self.root / "teacher_emb.fp16.memmap"
        if not meta_path.exists() or not index_path.exists() or not memmap_path.exists():
            raise FileNotFoundError(
                f"Teacher embeddings not found under {self.root}. Run 04_embed_teacher.py first."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.text_id_to_row = json.loads(index_path.read_text(encoding="utf-8"))
        self.dim = int(self.meta["dim"])
        self.n = int(self.meta["n"])
        self._mm_doc = np.memmap(
            memmap_path, dtype=np.float16, mode="r", shape=(self.n, self.dim)
        )
        qpath = self.root / self.meta.get("query_memmap", "teacher_emb_query.fp16.memmap")
        if qpath.exists():
            self._mm_query = np.memmap(
                qpath, dtype=np.float16, mode="r", shape=(self.n, self.dim)
            )
        else:
            self._mm_query = self._mm_doc

    def get(self, text_ids: list[str], use_query: list[bool] | None = None) -> torch.Tensor:
        rows = [self.text_id_to_row[tid] for tid in text_ids]
        if use_query is None:
            arr = np.asarray(self._mm_doc[rows], dtype=np.float32)
        else:
            vecs = []
            for row, qflag in zip(rows, use_query):
                src = self._mm_query if qflag else self._mm_doc
                vecs.append(np.asarray(src[row], dtype=np.float32))
            arr = np.stack(vecs, axis=0)
        return torch.from_numpy(arr)

    def __contains__(self, text_id: str) -> bool:
        return text_id in self.text_id_to_row


class ExampleShardDataset(Dataset):
    """Map-style dataset over parquet example shards."""

    def __init__(self, shards_root: str | Path, split: str = "train"):
        self.shards_root = Path(shards_root)
        pattern = f"examples-{split}-*.parquet"
        self.files = sorted(self.shards_root.glob(pattern))
        if not self.files:
            self.files = sorted(self.shards_root.glob("examples-*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No example shards in {self.shards_root}")
        self._frames: list[pd.DataFrame] = []
        self._cum: list[int] = []
        total = 0
        for f in self.files:
            df = pq.read_table(f).to_pandas()
            self._frames.append(df)
            total += len(df)
            self._cum.append(total)
        self._len = total

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)
        prev = 0
        for frame, cum in zip(self._frames, self._cum):
            if idx < cum:
                row = frame.iloc[idx - prev]
                return row.to_dict()
            prev = cum
        raise IndexError(idx)


def iter_texts_parquet(path: str | Path) -> Iterator[dict[str, Any]]:
    table = pq.read_table(path)
    df = table.to_pandas()
    for _, row in df.iterrows():
        yield row.to_dict()


def load_texts_dataframe(shards_root: str | Path) -> pd.DataFrame:
    path = Path(shards_root) / "texts.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pandas()
