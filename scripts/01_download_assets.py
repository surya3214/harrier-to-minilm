#!/usr/bin/env python3
"""Download teacher/student models and EN/KO datasets (internet host)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download
from datasets import load_dataset

from harrier_distill.config import load_yaml, project_root, resolve_path


def repo_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def download_model(repo_id: str, models_root: Path) -> Path:
    dest = models_root / repo_dirname(repo_id)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model {repo_id} -> {dest}")
    snapshot_download(repo_id=repo_id, local_dir=str(dest), local_dir_use_symlinks=False)
    return dest


def download_dataset_entry(entry: dict, datasets_root: Path) -> dict:
    name = entry["name"]
    config = entry.get("config")
    split = entry.get("split", "train")
    safe = name.replace("/", "__")
    if config:
        safe = f"{safe}__{config}"
    dest = datasets_root / safe
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset {name} config={config} split={split} -> {dest}")
    try:
        if config:
            ds = load_dataset(name, config, split=split)
        else:
            ds = load_dataset(name, split=split)
        # Optionally take a stream sample bound later in build; here save full requested split
        max_ex = entry.get("max_examples")
        if max_ex is not None and hasattr(ds, "select"):
            n = min(len(ds), int(max_ex) * 2)  # keep headroom; build script caps again
            ds = ds.select(range(n))
        out_path = dest / f"{split}.parquet"
        ds.to_parquet(str(out_path))
        return {
            "name": name,
            "config": config,
            "split": split,
            "lang": entry.get("lang"),
            "path": str(out_path.relative_to(project_root())),
            "rows": len(ds),
            "status": "ok",
        }
    except Exception as e:
        print(f"WARNING: failed to download {name} ({config}): {e}")
        return {
            "name": name,
            "config": config,
            "split": split,
            "lang": entry.get("lang"),
            "path": None,
            "rows": 0,
            "status": f"error: {e}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/en_ko_sts_retrieval.yaml")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    models_root = resolve_path(cfg, "models_root", root)
    datasets_root = resolve_path(cfg, "datasets_root", root)
    models_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "languages": cfg.get("languages", {}).get("active", []),
        "tasks": cfg.get("tasks", {}).get("active", []),
        "models": {},
        "datasets": [],
    }

    if not args.skip_models:
        for key in ("teacher", "student"):
            repo = cfg["models"][key]
            path = download_model(repo, models_root)
            manifest["models"][key] = {
                "repo_id": repo,
                "path": str(path.relative_to(root)),
            }

    if not args.skip_datasets:
        for group in ("sts", "retrieval", "bitext"):
            for entry in cfg.get("datasets", {}).get(group, []):
                info = download_dataset_entry(entry, datasets_root)
                info["group"] = group
                manifest["datasets"].append(info)

    manifest_path = resolve_path(cfg, "manifest", root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
