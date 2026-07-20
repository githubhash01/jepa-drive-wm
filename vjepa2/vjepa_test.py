import torch
from pathlib import Path

ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "vjepa2-ac-vitg.pt"
print("checkpoint:", ckpt)
print("exists:", ckpt.exists())
print("size GB:", ckpt.stat().st_size / 1e9)

encoder, predictor = torch.hub.load(
    "/home/hashim/PLSLAM/vjepa2",
    "vjepa2_ac_vit_giant",
    source="local",
    trust_repo=True,
)

encoder.eval()
predictor.eval()

print("loaded")
print("patch_size:", encoder.patch_size)
