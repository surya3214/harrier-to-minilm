#!/usr/bin/env python3
"""Held-out EN/KO STS Spearman + small retrieval proxy (offline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harrier_distill.config import load_yaml, project_root, resolve_path
from harrier_distill.data import ExampleShardDataset
from harrier_distill.performance import apply_torch_performance, move_batch
from harrier_distill.prompts import PromptMap
from harrier_distill.student_model import StudentEncoder


@torch.inference_mode()
def encode(model: StudentEncoder, texts: list[str], device, max_bs: int = 64) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), max_bs):
        chunk = texts[i : i + max_bs]
        tok = model.tokenize(chunk)
        tok = move_batch(tok, device, non_blocking=True)
        emb = model(tok["input_ids"], tok["attention_mask"])
        embs.append(emb.float().cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, model.encoder.config.hidden_size))


def eval_sts(model, prompt_map, examples, device, lang_filter: str | None = None):
    pairs_a, pairs_b, labels = [], [], []
    for ex in examples:
        if ex.get("task") != "sts":
            continue
        if lang_filter and not str(ex.get("lang", "")).startswith(lang_filter):
            continue
        texts = ex["texts"] if isinstance(ex["texts"], list) else list(ex["texts"])
        if len(texts) < 2:
            continue
        label = ex.get("label")
        if label is None:
            continue
        s_prompts = (
            ex["student_prompt_names"]
            if isinstance(ex["student_prompt_names"], list)
            else list(ex["student_prompt_names"])
        )
        a = prompt_map.student_text(s_prompts[0], str(texts[0]))
        # second sentence: document / raw
        b = prompt_map.student_text("document", str(texts[1]))
        pairs_a.append(a)
        pairs_b.append(b)
        labels.append(float(label))
    if len(labels) < 3:
        return {"n": len(labels), "spearman": None}
    ea = encode(model, pairs_a, device)
    eb = encode(model, pairs_b, device)
    sims = np.sum(ea * eb, axis=1)
    corr, _ = spearmanr(sims, np.asarray(labels))
    return {"n": len(labels), "spearman": float(corr)}


def eval_retrieval_proxy(model, prompt_map, examples, device, limit: int = 512):
    """In-batch style: each query ranks its positive among collected passages."""
    queries, q_langs = [], []
    passages, p_owner = [], []
    count = 0
    for ex in examples:
        if ex.get("task") != "retrieval":
            continue
        texts = ex["texts"] if isinstance(ex["texts"], list) else list(ex["texts"])
        roles = ex["roles"] if isinstance(ex["roles"], list) else list(ex["roles"])
        if "query" not in roles or "positive" not in roles:
            continue
        q = texts[roles.index("query")]
        p = texts[roles.index("positive")]
        queries.append(prompt_map.student_text("ret", str(q)))
        q_langs.append(str(ex.get("lang", "en")))
        passages.append(prompt_map.student_text("document", str(p)))
        p_owner.append(count)
        count += 1
        if count >= limit:
            break
    if count < 2:
        return {"n": count, "recall@1": None, "mrr": None}

    eq = encode(model, queries, device)
    ep = encode(model, passages, device)
    scores = eq @ ep.T
    hits = 0
    mrr = 0.0
    for i in range(count):
        order = np.argsort(-scores[i])
        rank = int(np.where(order == i)[0][0]) + 1
        if rank == 1:
            hits += 1
        mrr += 1.0 / rank
    return {
        "n": count,
        "recall@1": hits / count,
        "mrr": mrr / count,
        "by_lang": {
            lang: int(sum(1 for x in q_langs if x == lang)) for lang in sorted(set(q_langs))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/en_ko_sts_retrieval.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Student checkpoint dir (default: outputs/student_final or base student)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    perf = cfg.get("performance", {})
    apply_torch_performance(perf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_root = resolve_path(cfg, "outputs_root", root)
    shards_root = resolve_path(cfg, "shards_root", root)

    if args.model:
        model_path = Path(args.model)
    elif (outputs_root / "student_final").exists():
        model_path = outputs_root / "student_final"
    else:
        model_path = root / cfg["models"]["student_local"]
        if not model_path.exists():
            model_path = Path(cfg["models"]["student"])

    print(f"Evaluating {model_path} on device={device}")
    model = StudentEncoder(model_path, max_length=int(cfg.get("max_length", 512)), perf=perf)
    model.to(device)
    model.eval()
    prompt_map = PromptMap.from_config(cfg)

    # Prefer val shards; fall back to train head
    try:
        ds = ExampleShardDataset(shards_root, split="val")
    except FileNotFoundError:
        ds = ExampleShardDataset(shards_root, split="train")
    examples = [ds[i] for i in range(len(ds))]

    results = {
        "model": str(model_path),
        "sts_en": eval_sts(model, prompt_map, examples, device, lang_filter="en"),
        "sts_ko": eval_sts(model, prompt_map, examples, device, lang_filter="ko"),
        "retrieval_proxy": eval_retrieval_proxy(
            model,
            prompt_map,
            examples,
            device,
            limit=int(cfg.get("eval", {}).get("retrieval_proxy_size", 512)),
        ),
    }
    out_path = outputs_root / "smoke_eval.json"
    outputs_root.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
