#!/usr/bin/env python3
"""Normalize downloaded datasets into texts.parquet + example shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harrier_distill.config import load_yaml, project_root, resolve_path
from harrier_distill.schema import make_example


def text_id_for(text: str, lang: str) -> str:
    h = hashlib.sha1(f"{lang}||{text}".encode("utf-8")).hexdigest()[:16]
    return f"{lang}_{h}"


def read_local_parquet(datasets_root: Path, name: str, config: str | None, split: str) -> pd.DataFrame | None:
    safe = name.replace("/", "__")
    if config:
        safe = f"{safe}__{config}"
    path = datasets_root / safe / f"{split}.parquet"
    if not path.exists():
        print(f"  skip missing {path}")
        return None
    return pq.read_table(path).to_pandas()


def iter_sts(df: pd.DataFrame, lang: str, name: str) -> Iterator[dict[str, Any]]:
    # Common column variants
    cols = set(df.columns)
    if {"sentence1", "sentence2"}.issubset(cols):
        a_col, b_col = "sentence1", "sentence2"
    elif {"sentence_a", "sentence_b"}.issubset(cols):
        a_col, b_col = "sentence_a", "sentence_b"
    elif {"text1", "text2"}.issubset(cols):
        a_col, b_col = "text1", "text2"
    elif {"premise", "hypothesis"}.issubset(cols):
        a_col, b_col = "premise", "hypothesis"
    else:
        print(f"  unrecognized STS columns for {name}: {cols}")
        return

    score_col = None
    for c in ("score", "label", "similarity"):
        if c in cols:
            score_col = c
            break

    for i, row in df.iterrows():
        a, b = str(row[a_col]).strip(), str(row[b_col]).strip()
        if not a or not b:
            continue
        label = float(row[score_col]) if score_col is not None else None
        # NLI labels: keep entailment-ish as positives via numeric if present
        yield make_example(
            example_id=f"sts-{lang}-{name}-{i}",
            task="sts",
            lang=lang,
            texts=[a, b],
            roles=["sentence_a", "sentence_b"],
            teacher_prompt_names=["sts_query", "document"],
            student_prompt_names=["sts", "document"],
            label=label,
        )


def iter_retrieval_msmarco(df: pd.DataFrame, max_neg: int) -> Iterator[dict[str, Any]]:
    cols = set(df.columns)
    # sentence-transformers msmarco-hard-negatives style
    q_col = "query" if "query" in cols else None
    pos_col = "positive" if "positive" in cols else ("pos" if "pos" in cols else None)
    neg_col = "negative" if "negative" in cols else ("neg" if "neg" in cols else None)
    if not q_col:
        print(f"  unrecognized MS MARCO columns: {cols}")
        return
    for i, row in df.iterrows():
        q = str(row[q_col]).strip()
        if not q:
            continue
        texts = [q]
        roles = ["query"]
        t_prompts: list[str | None] = ["web_search_query"]
        s_prompts: list[str | None] = ["ret"]

        pos = row[pos_col] if pos_col else None
        if isinstance(pos, list):
            pos_list = [str(x) for x in pos[:1]]
        elif pos is not None and str(pos).strip():
            pos_list = [str(pos)]
        else:
            continue
        for p in pos_list:
            texts.append(p)
            roles.append("positive")
            t_prompts.append("document")
            s_prompts.append("document")

        negs = row[neg_col] if neg_col else []
        if isinstance(negs, list):
            neg_list = [str(x) for x in negs[:max_neg]]
        elif negs is not None and str(negs).strip():
            neg_list = [str(negs)]
        else:
            neg_list = []
        for n in neg_list:
            if not n.strip():
                continue
            texts.append(n)
            roles.append("negative")
            t_prompts.append("document")
            s_prompts.append("document")

        yield make_example(
            example_id=f"ret-en-msmarco-{i}",
            task="retrieval",
            lang="en",
            texts=texts,
            roles=roles,
            teacher_prompt_names=t_prompts,
            student_prompt_names=s_prompts,
        )


def iter_retrieval_miracl(df: pd.DataFrame, lang: str, max_neg: int) -> Iterator[dict[str, Any]]:
    cols = set(df.columns)
    # MIRACL train often has query_id, query, positive_passages, negative_passages
    if "query" not in cols:
        print(f"  unrecognized MIRACL columns: {cols}")
        return
    for i, row in df.iterrows():
        q = str(row["query"]).strip()
        if not q:
            continue
        texts = [q]
        roles = ["query"]
        t_prompts: list[str | None] = ["web_search_query"]
        s_prompts: list[str | None] = ["ret"]

        pos = row.get("positive_passages", row.get("positive", []))
        if isinstance(pos, list):
            for p in pos[:1]:
                if isinstance(p, dict):
                    ptext = str(p.get("text") or p.get("passage") or "").strip()
                else:
                    ptext = str(p).strip()
                if ptext:
                    texts.append(ptext)
                    roles.append("positive")
                    t_prompts.append("document")
                    s_prompts.append("document")
        if len(texts) < 2:
            continue

        negs = row.get("negative_passages", row.get("negative", []))
        count = 0
        if isinstance(negs, list):
            for n in negs:
                if count >= max_neg:
                    break
                if isinstance(n, dict):
                    ntext = str(n.get("text") or n.get("passage") or "").strip()
                else:
                    ntext = str(n).strip()
                if ntext:
                    texts.append(ntext)
                    roles.append("negative")
                    t_prompts.append("document")
                    s_prompts.append("document")
                    count += 1

        yield make_example(
            example_id=f"ret-{lang}-miracl-{i}",
            task="retrieval",
            lang=lang,
            texts=texts,
            roles=roles,
            teacher_prompt_names=t_prompts,
            student_prompt_names=s_prompts,
        )


def iter_retrieval_mrtydi(df: pd.DataFrame, lang: str) -> Iterator[dict[str, Any]]:
    cols = set(df.columns)
    q_col = "query" if "query" in cols else ("question" if "question" in cols else None)
    if not q_col:
        print(f"  unrecognized Mr.TyDi columns: {cols}")
        return
    for i, row in df.iterrows():
        q = str(row[q_col]).strip()
        pos = row.get("positive_passages", row.get("positive_passage", row.get("answers")))
        ptext = None
        if isinstance(pos, list) and pos:
            p0 = pos[0]
            ptext = str(p0.get("text") if isinstance(p0, dict) else p0).strip()
        elif isinstance(pos, dict):
            ptext = str(pos.get("text") or "").strip()
        elif pos is not None:
            ptext = str(pos).strip()
        if not q or not ptext:
            continue
        yield make_example(
            example_id=f"ret-{lang}-mrtydi-{i}",
            task="retrieval",
            lang=lang,
            texts=[q, ptext],
            roles=["query", "positive"],
            teacher_prompt_names=["web_search_query", "document"],
            student_prompt_names=["ret", "document"],
        )


def iter_bitext(df: pd.DataFrame, lang: str) -> Iterator[dict[str, Any]]:
    cols = set(df.columns)
    if {"sentence1", "sentence2"}.issubset(cols):
        a_col, b_col = "sentence1", "sentence2"
    elif {"english", "korean"}.issubset(cols):
        a_col, b_col = "english", "korean"
    elif {"en", "ko"}.issubset(cols):
        a_col, b_col = "en", "ko"
    elif {"text1", "text2"}.issubset(cols):
        a_col, b_col = "text1", "text2"
    else:
        # try first two string columns
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if len(str_cols) < 2:
            print(f"  unrecognized bitext columns: {cols}")
            return
        a_col, b_col = str_cols[0], str_cols[1]

    for i, row in df.iterrows():
        a, b = str(row[a_col]).strip(), str(row[b_col]).strip()
        if not a or not b:
            continue
        yield make_example(
            example_id=f"bitext-{lang}-{i}",
            task="bitext",
            lang=lang,
            texts=[a, b],
            roles=["sentence_a", "sentence_b"],
            teacher_prompt_names=["bitext_query", "bitext_query"],
            student_prompt_names=["bitext", "bitext"],
        )


def assign_text_ids(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    text_map: dict[str, dict[str, str]] = {}
    out = []
    for ex in examples:
        lang = ex["lang"].split("-")[0] if "-" in ex["lang"] else ex["lang"]
        tids = []
        for t in ex["texts"]:
            # for bitext en-ko, tag by side index language loosely
            tid = text_id_for(t, lang)
            tids.append(tid)
            text_map[tid] = {"text_id": tid, "text": t, "lang": lang}
        ex = {**ex, "text_ids": tids}
        out.append(ex)
    return out, text_map


def write_shards(examples: list[dict[str, Any]], shards_root: Path, per_shard: int) -> None:
    shards_root.mkdir(parents=True, exist_ok=True)
    # hold out 1% for smoke eval if enough data
    n = len(examples)
    n_val = max(1, n // 100) if n >= 100 else 0
    train = examples[n_val:]
    val = examples[:n_val]

    def _write(split: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        for si, start in enumerate(range(0, len(rows), per_shard)):
            chunk = rows[start : start + per_shard]
            df = pd.DataFrame(chunk)
            path = shards_root / f"examples-{split}-{si:05d}.parquet"
            pq.write_table(pa.Table.from_pandas(df), path)
            print(f"  wrote {path} ({len(chunk)} rows)")

    _write("train", train)
    _write("val", val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/en_ko_sts_retrieval.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    datasets_root = resolve_path(cfg, "datasets_root", root)
    shards_root = resolve_path(cfg, "shards_root", root)
    shards_root.mkdir(parents=True, exist_ok=True)

    examples: list[dict[str, Any]] = []

    # STS
    for entry in cfg.get("datasets", {}).get("sts", []):
        df = read_local_parquet(datasets_root, entry["name"], entry.get("config"), entry.get("split", "train"))
        if df is None:
            continue
        max_ex = int(entry.get("max_examples", len(df)))
        df = df.head(max_ex)
        print(f"Building STS from {entry['name']} ({len(df)})")
        examples.extend(list(iter_sts(df, entry.get("lang", "en"), entry["name"].replace("/", "_"))))

    # Retrieval
    for entry in cfg.get("datasets", {}).get("retrieval", []):
        df = read_local_parquet(datasets_root, entry["name"], entry.get("config"), entry.get("split", "train"))
        if df is None:
            continue
        max_ex = int(entry.get("max_examples", len(df)))
        df = df.head(max_ex)
        max_neg = int(entry.get("max_negatives", 3))
        name = entry["name"]
        lang = entry.get("lang", "en")
        print(f"Building retrieval from {name} lang={lang} ({len(df)})")
        if "msmarco" in name:
            examples.extend(list(iter_retrieval_msmarco(df, max_neg)))
        elif "miracl" in name:
            examples.extend(list(iter_retrieval_miracl(df, lang, max_neg)))
        elif "mr-tydi" in name or "mr_tydi" in name:
            examples.extend(list(iter_retrieval_mrtydi(df, lang)))
        else:
            examples.extend(list(iter_retrieval_miracl(df, lang, max_neg)))

    # Bitext
    for entry in cfg.get("datasets", {}).get("bitext", []):
        df = read_local_parquet(datasets_root, entry["name"], entry.get("config"), entry.get("split", "train"))
        if df is None:
            continue
        max_ex = int(entry.get("max_examples", len(df)))
        df = df.head(max_ex)
        print(f"Building bitext from {entry['name']} ({len(df)})")
        examples.extend(list(iter_bitext(df, entry.get("lang", "en-ko"))))

    if not examples:
        print("WARNING: no examples built. Creating tiny synthetic EN/KO smoke shards for pipeline testing.")
        examples = [
            make_example(
                "syn-en-0",
                "sts",
                "en",
                ["A dog runs in the park.", "A puppy is running outdoors."],
                ["sentence_a", "sentence_b"],
                ["sts_query", "document"],
                ["sts", "document"],
                label=4.0,
            ),
            make_example(
                "syn-ko-0",
                "sts",
                "ko",
                ["개가 공원에서 달린다.", "강아지가 밖에서 뛰고 있다."],
                ["sentence_a", "sentence_b"],
                ["sts_query", "document"],
                ["sts", "document"],
                label=4.0,
            ),
            make_example(
                "syn-ret-0",
                "retrieval",
                "en",
                ["what is protein", "Protein is an essential nutrient made of amino acids."],
                ["query", "positive"],
                ["web_search_query", "document"],
                ["ret", "document"],
            ),
            make_example(
                "syn-bitext-0",
                "bitext",
                "en-ko",
                ["Hello world", "안녕 세계"],
                ["sentence_a", "sentence_b"],
                ["bitext_query", "bitext_query"],
                ["bitext", "bitext"],
            ),
        ]

    examples, text_map = assign_text_ids(examples)
    max_unique = int(cfg.get("caps", {}).get("max_unique_texts", 2_000_000))
    if len(text_map) > max_unique:
        # keep first max_unique text ids and filter examples
        keep = set(list(text_map.keys())[:max_unique])
        text_map = {k: v for k, v in text_map.items() if k in keep}
        filtered = []
        for ex in examples:
            if all(tid in keep for tid in ex["text_ids"]):
                filtered.append(ex)
        examples = filtered

    texts_df = pd.DataFrame(list(text_map.values()))
    texts_path = shards_root / "texts.parquet"
    pq.write_table(pa.Table.from_pandas(texts_df), texts_path)
    print(f"Wrote {texts_path} ({len(texts_df)} unique texts)")

    per_shard = int(cfg.get("caps", {}).get("examples_per_shard", 100_000))
    write_shards(examples, shards_root, per_shard)

    meta = {
        "n_examples": len(examples),
        "n_texts": len(texts_df),
        "languages": sorted({ex["lang"] for ex in examples}),
        "tasks": sorted({ex["task"] for ex in examples}),
    }
    (shards_root / "shards_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
