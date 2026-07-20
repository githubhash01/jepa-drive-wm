import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def clean_backbone_key(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("backbone.", "")
        cleaned[k] = v
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        default="vjepa2_1_vitG_384.pt",
        help="Path to V-JEPA 2.1 checkpoint",
    )
    parser.add_argument(
        "--out",
        default="image_embedding.pt",
        help="Where to save the output tensor",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda, cpu, or mps",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model on {device}...")

    # Load architecture from the local cloned repo.
    # This returns (encoder, predictor); for encoding an image we only need encoder.
    encoder, _ = torch.hub.load(
        ".",
        "vjepa2_1_vit_gigantic_384",
        source="local",
        pretrained=False,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # For V-JEPA 2.1 ViT-g / ViT-G, the hub loader uses target_encoder by default.
    # Some checkpoints may also contain ema_encoder, so we fallback gracefully.
    if "target_encoder" in ckpt:
        encoder_state = ckpt["target_encoder"]
        print("Using checkpoint key: target_encoder")
    elif "ema_encoder" in ckpt:
        encoder_state = ckpt["ema_encoder"]
        print("Using checkpoint key: ema_encoder")
    elif "encoder" in ckpt:
        encoder_state = ckpt["encoder"]
        print("Using checkpoint key: encoder")
    else:
        raise KeyError(f"Could not find encoder weights. Available keys: {list(ckpt.keys())}")

    encoder_state = clean_backbone_key(encoder_state)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    encoder.eval()
    encoder.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(
                (384, 384),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)  # [1, 3, 384, 384]

    print(f"Input tensor: {tuple(x.shape)}")

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                patch_tokens = encoder(x)
        else:
            patch_tokens = encoder(x)

    # For 384 / 16 patches: 24 x 24 = 576 patch tokens.
    # ViT-Gigantic hidden dim should be 1664.
    print(f"Patch tokens: {tuple(patch_tokens.shape)}")

    # Global image embedding by mean-pooling patch tokens.
    image_embedding = patch_tokens.mean(dim=1)
    print(f"Global embedding: {tuple(image_embedding.shape)}")

    output = {
        "patch_tokens": patch_tokens.detach().cpu(),
        "image_embedding": image_embedding.detach().cpu(),
    }

    torch.save(output, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()