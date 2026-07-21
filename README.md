# VJEPA-2.1 Driving World Model

Thesis codebase: a V-JEPA 2.1 latent world model plus dense decoders (depth,
semantic segmentation) trained on pseudolabeled KITTI odometry.

## Layout

```
src/jepa_drive_wm/     the package (data, world_model, dense_decoder, utils, viz, ...)
vjepa2/                vendored V-JEPA 2 checkout (code tracked; checkpoints git-ignored)
dataset/KITTI/         KITTI odometry root (git-ignored; symlink or real data)
outputs/               trained checkpoints, PCA files (git-ignored)
scripts/migrate/       machine-migration helpers (pack_usb / unpack_uni / build_latents_all)
```

All paths are centralised in `src/jepa_drive_wm/paths.py` and overridable via
`JEPA_KITTI_ROOT`, `JEPA_VJEPA_REPO`, `JEPA_VJEPA_CKPT_DIR`, `JEPA_OUTPUTS_DIR`.
Defaults are relative to the repo root, so a clone with data under
`dataset/KITTI` needs no configuration.

## Setup on a new machine

1. **Clone**
   ```bash
   git clone git@github.com:githubhash01/jepa-drive-wm.git && cd jepa-drive-wm
   ```
2. **Environment** (Python 3.12; torch first, matched to the local CUDA)
   ```bash
   conda create -n vjepa2_env python=3.12 -y && conda activate vjepa2_env
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130  # pick your CUDA
   pip install -r requirements.txt
   pip install -e .
   ```
3. **Weights & Biases** — all training logs to the `jepa-drive-wm` project, so
   authenticate before launching (an unauthenticated run started under
   `tmux`/`nohup` will hang on the interactive login prompt):
   ```bash
   wandb login                    # paste the key from https://wandb.ai/authorize
   # headless alternative: export WANDB_API_KEY=<key>   (add to your shell rc)
   # no network? export WANDB_MODE=offline, then `wandb sync wandb/latest-run` later
   ```
4. **Data** — from the transfer USB stick:
   ```bash
   ./scripts/migrate/unpack_uni.sh /media/<user>/<STICK>/jepa-drive-wm-transfer "$PWD"
   ```
   This lays KITTI (images, depth + semantic pseudolabels, poses) into
   `dataset/KITTI/` and the ViT-B checkpoint into `vjepa2/model_checkpoints/vjepa21/`.
5. **Regenerate V-JEPA latents** (not transferred — 117G, rebuilt on the GPU):
   ```bash
   ./scripts/migrate/build_latents_all.sh 4      # smoke test: smallest sequence (271 frames)
   ./scripts/migrate/build_latents_all.sh        # all sequences 0..21 (~43.5k frames)
   ```
6. **Smoke tests**
   ```bash
   PYTHONPATH=src python -m jepa_drive_wm.data.kitti                          # calib sanity
   PYTHONPATH=src python -m jepa_drive_wm.dense_decoder.data_interface_dense  # dense batches
   PYTHONPATH=src python -m jepa_drive_wm.world_model.data_interface_wm       # rollout batches
   PYTHONPATH=src python -m jepa_drive_wm.world_model.train_wm --smoke-test   # full loop
   ```

## Training

```bash
PYTHONPATH=src python -m jepa_drive_wm.world_model.train_wm
PYTHONPATH=src python -m jepa_drive_wm.dense_decoder.train_depth
PYTHONPATH=src python -m jepa_drive_wm.dense_decoder.train_semantics
```

Checkpoints land in `outputs/`; runs log to wandb project `jepa-drive-wm`.

## Notes

- Depth pseudolabels came from FoundationStereo, semantics from OneFormer
  (`utils/depth_builder.py`, `utils/oneformer_kitti.py`). Both are already built
  and travel with the dataset; the generators (and FoundationStereo itself) are
  not needed on the training machine.
- Only the ViT-B distilled checkpoint (`vjepa2_1_vitb_dist_vitG_384.pt`) is used.
