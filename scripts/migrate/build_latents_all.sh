#!/usr/bin/env bash
# Regenerate the per-frame V-JEPA ViT-B latent caches (vjepa_vitb/) for all
# KITTI sequences on the GPU. ~43.5k frames total; ~117G of .npy output.
#
#   ./scripts/migrate/build_latents_all.sh          # all sequences 0..21
#   ./scripts/migrate/build_latents_all.sh 4        # just sequence 04 (smallest — smoke test)
#
# Run from the repo root inside the conda env. The builder is resumable
# per-sequence: already-written frames are skipped on re-run.
set -euo pipefail
cd "$(dirname "$0")/../.."

SEQS="${1:-0-21}"
PYTHONPATH=src python -m jepa_drive_wm.utils.vjepa_embeddings_builder --sequences "$SEQS"
