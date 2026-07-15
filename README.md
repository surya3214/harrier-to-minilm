# Harrier → mMiniLMv2 Distillation (EN/KO)

Air-gap-friendly pipeline: download on an internet host, transfer offline artifacts, then embed + train on a GPU host without network access.

## Models

| Role | Model |
|---|---|
| Teacher | `microsoft/harrier-oss-v1-0.6b` |
| Student | `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large` |

- Max sequence length: **512**
- Teacher prompts: full Harrier instruct templates
- Student prompts: minimized tags (`sts:`, `ret:`, `bitext:`)
- Losses: **pad-safe KL similarity-matrix distillation** + **focal InfoNCE** + **light disperse** + **STS pair cosine MSE**
- FlashAttention-2: **disabled by default**; `pin_memory`, fused AdamW, TF32, bf16, grad checkpointing enabled
- STS roles: teacher query prompt on `sentence_a` only; `sentence_b` uses document embeddings (same as student)

## Layout

```text
configs/en_ko_sts_retrieval.yaml
configs/languages/registry.yaml      # later language stubs
configs/tasks/registry.yaml          # clustering/classification/reranking stubs
src/harrier_distill/
scripts/01_download_assets.py        # internet host
scripts/02_build_shards.py           # internet host
scripts/03_pack_transfer.sh          # internet host
scripts/04_embed_teacher.py          # GPU host (multi-GPU)
scripts/05_train_student.py          # GPU host (multi-GPU)
scripts/06_smoke_eval.py             # GPU host
```

## Air-gap runbook

### A) Internet host (no GPU required)

```bash
pip install -r requirements.txt
pip install -e .

python scripts/01_download_assets.py --config configs/en_ko_sts_retrieval.yaml
python scripts/02_build_shards.py --config configs/en_ko_sts_retrieval.yaml
bash scripts/03_pack_transfer.sh
```

Copy `transfer/harrier_distill_en_ko_v1_*.tar` to the GPU host.

If dataset downloads fail for some sources, `02_build_shards.py` still emits tiny synthetic EN/KO shards so the GPU path can be validated.

### B) GPU host (no internet)

```bash
tar -xf harrier_distill_en_ko_v1_*.tar
pip install -r requirements.txt   # use local/vendor wheels if air-gapped
pip install -e .

# Teacher embeddings (multi-GPU)
torchrun --nproc_per_node=8 scripts/04_embed_teacher.py --config configs/en_ko_sts_retrieval.yaml

# Student training (multi-GPU)
torchrun --nproc_per_node=8 scripts/05_train_student.py --config configs/en_ko_sts_retrieval.yaml

# Smoke metrics (STS Spearman EN/KO + retrieval proxy)
python scripts/06_smoke_eval.py --config configs/en_ko_sts_retrieval.yaml
```

Single-GPU works without `torchrun` (plain `python scripts/04_...` / `05_...`).

## Retrain note (loss fix)

After pulling pad-safe / STS-MSE training changes, **retrain the student** (teacher memmaps do not need re-embedding unless you change prompt policy). Prior checkpoints trained without pad masking + STS MSE are not comparable.

Suggested short validation:

```bash
torchrun --nproc_per_node=$N scripts/05_train_student.py \
  --config configs/en_ko_sts_retrieval.yaml --max-steps 200
python scripts/06_smoke_eval.py --config configs/en_ko_sts_retrieval.yaml
```

Watch `sts=` in the train log — it should drop; held-out STS Spearman should rise vs the previous plateau.

## Performance knobs

See `performance:` in [`configs/en_ko_sts_retrieval.yaml`](configs/en_ko_sts_retrieval.yaml):

- `flash_attention_2: false` (opt-in)
- `pin_memory`, `persistent_workers`, `non_blocking`, `tf32`, `cudnn_benchmark`, `fused_adamw`, `bf16`, `gradient_checkpointing`: on by default

## Later expansion

- Languages: add entries under [`configs/languages/registry.yaml`](configs/languages/registry.yaml) and dataset lists in the main config
- Tasks: enable `clustering` / `classification` / `reranking` stubs in [`configs/tasks/registry.yaml`](configs/tasks/registry.yaml) (reranking will add Margin-MSE later; shard schema is already shared)

## v1 data scope

EN + KO for **STS**, **Retrieval**, and **EN-KO bitext**. Unique texts from training pairs/triples only (no full MIRACL corpus dump).
