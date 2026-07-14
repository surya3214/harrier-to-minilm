#!/usr/bin/env bash
# Pack models + shards + configs for offline transfer to the GPU host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-configs/en_ko_sts_retrieval.yaml}"
TRANSFER_DIR="${ROOT}/transfer"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_TAR="${TRANSFER_DIR}/harrier_distill_en_ko_v1_${STAMP}.tar"

mkdir -p "${TRANSFER_DIR}"
cd "${ROOT}"

echo "Packing transfer archive -> ${OUT_TAR}"

# Base code + configs (always)
tar -cf "${OUT_TAR}" \
  configs \
  requirements.txt \
  pyproject.toml \
  src \
  scripts \
  README.md

# Optional heavy artifacts
if [[ -d artifacts/models ]]; then
  tar -rf "${OUT_TAR}" artifacts/models
fi
if [[ -d artifacts/shards ]]; then
  tar -rf "${OUT_TAR}" artifacts/shards
fi
if [[ -f artifacts/MANIFEST.json ]]; then
  tar -rf "${OUT_TAR}" artifacts/MANIFEST.json
fi
# Optionally include raw hf datasets (large) — only if explicitly requested
if [[ "${INCLUDE_DATASETS:-0}" == "1" ]] && [[ -d artifacts/hf_datasets ]]; then
  tar -rf "${OUT_TAR}" artifacts/hf_datasets
fi

echo "Checksums:"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${OUT_TAR}" | tee "${OUT_TAR}.sha256"
else
  shasum -a 256 "${OUT_TAR}" | tee "${OUT_TAR}.sha256"
fi

ls -lh "${OUT_TAR}"
echo "Done. Copy ${OUT_TAR} to the GPU host and extract at the project root."
echo "Config used: ${CONFIG}"
