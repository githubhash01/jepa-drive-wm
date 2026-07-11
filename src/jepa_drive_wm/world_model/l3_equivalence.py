"""L3-state architectural equivalence check.

The world-model direction predicts the future of a scene in V-JEPA 2.1 feature
space. The candidate **state** is the *raw residual-stream* output after the 3rd
transformer block of ViT-B/16 (0-indexed block index 2 — the first hierarchical
tap). A future WM predicts x_hat_{t+1}^{(3)} and pushes it through the frozen
remainder of the encoder (blocks 4-12) to regenerate the L6/L9/L12 hierarchy the
DPT probes consume.

Before training that WM is worth attempting, one invariant must hold:

    Tail_{4:12}(x^{(3)}) == FullEncoder(I)   for L6/L9/L12, up to fp error.

This module builds exactly that go/no-go check and nothing else:

  1. ``capture_raw_l3`` — a forward hook on ``blocks[2]`` grabs the un-normed
     residual stream (the existing hierarchical path only returns *normed* taps).
  2. ``Tail`` — reruns ``blocks[3:12]`` with the right geometry (RoPE needs
     H/W_patches) and re-applies ``norms_block`` at taps 5/8/11 + the final norm,
     reproducing the normed L6/L9/L12 the probe already uses.
  3. ``run_equivalence`` — compares reconstructed vs reference in fp32 (hard gate)
     and bf16 (reported error floor), plus a normed-L3 negative control.

Run:  python -m jepa_drive_wm.world_model.l3_equivalence
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch

from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Wrapper, VJEPA21Size


# ----------------------------------------------------------------------
# Raw-L3 capture (un-normed residual stream after the first tapped block)
# ----------------------------------------------------------------------
def _first_tap(encoder: torch.nn.Module) -> int:
    """0-indexed block whose *output* is the L3 state (first hierarchical tap)."""
    return int(encoder.hierarchical_layers[0])


def capture_raw_l3(wrapper: VJEPA21Wrapper, batch_chw: torch.Tensor) -> torch.Tensor:
    """Raw residual stream after the first tapped block, ``(B, N, D)``.

    Runs a single normal forward pass (so RoPE / patch-embed / eval / dtype are
    exactly as in production) and captures ``blocks[tap]``'s output via a forward
    hook. The block returns ``(x, attn)``; we grab ``x`` before any ``norms_block``
    is applied — this is what the existing ``out_layers`` path does *not* expose.
    """
    encoder = wrapper.encoder
    tap = _first_tap(encoder)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        x = output[0] if isinstance(output, (tuple, list)) else output
        captured["l3"] = x.detach().clone()

    handle = encoder.blocks[tap].register_forward_hook(hook)
    try:
        x = batch_chw.to(
            wrapper.device,
            wrapper.compute_dtype if wrapper.device.type == "cuda" else None,
        )
        x = x.unsqueeze(2)  # (B,C,H,W) -> (B,C,1,H,W) routes to the image patch-embed
        with torch.no_grad():
            if wrapper.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=wrapper.compute_dtype):
                    encoder(x)
            else:
                encoder(x)
    finally:
        handle.remove()

    if "l3" not in captured:
        raise RuntimeError("block-2 forward hook did not fire; encoder structure changed?")
    return captured["l3"]


# ----------------------------------------------------------------------
# Frozen tail: reinsert raw L3 and run blocks 4-12
# ----------------------------------------------------------------------
class Tail(torch.nn.Module):
    """Runs ``encoder.blocks[tap+1:]`` over a raw-L3 state and re-norms the taps.

    Reuses the *same* encoder object (no weight copy) so this is guaranteed to be
    the identical computation the full forward pass performs on blocks 4-12.
    Returns a dict of the normed later taps, keyed by 0-indexed block number, e.g.
    ``{5: L6, 8: L9, 11: L12}`` for ViT-B — matching ``outs[1:]`` of the
    hierarchical forward.
    """

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.tap = _first_tap(encoder)
        self.hier = list(encoder.hierarchical_layers)  # [2,5,8,11]
        self.n_blocks = len(encoder.blocks)

    def forward(
        self,
        l3: torch.Tensor,
        *,
        T: int,
        H_patches: int,
        W_patches: int,
        mode: str = "img",
    ) -> dict[int, torch.Tensor]:
        enc = self.encoder
        x = l3
        outs: dict[int, torch.Tensor] = {}
        for i in range(self.tap + 1, self.n_blocks):
            x, _ = enc.blocks[i](
                x,
                mask=None,
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                return_attn=False,
                mode=mode,
            )
            if i in self.hier:
                norm = enc.norms_block[self.hier.index(i)]
                outs[i] = norm(x)
        # Final-block tap already covered above (last hier entry == last block),
        # but guard the case where the last block is not a tap.
        if (self.n_blocks - 1) not in outs:
            outs[self.n_blocks - 1] = enc.norms_block[-1](x)
        return outs


# ----------------------------------------------------------------------
# Comparison utilities
# ----------------------------------------------------------------------
def _errors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1
    )
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "cos_min": cos.min().item(),
    }


@contextlib.contextmanager
def _as_dtype(encoder: torch.nn.Module, dtype: Optional[torch.dtype]):
    """Temporarily cast the encoder to ``dtype`` (no-op if ``dtype is None``)."""
    if dtype is None:
        yield
        return
    orig = next(encoder.parameters()).dtype
    encoder.to(dtype)
    try:
        yield
    finally:
        encoder.to(orig)


def _reference_hierarchical(wrapper: VJEPA21Wrapper, batch_chw: torch.Tensor) -> dict[int, torch.Tensor]:
    """Full-forward normed taps, keyed by 0-indexed block: {2:L3, 5:L6, 8:L9, 11:L12}."""
    wrapper._enable_hierarchical()
    encoder = wrapper.encoder
    x = batch_chw.to(
        wrapper.device, wrapper.compute_dtype if wrapper.device.type == "cuda" else None
    ).unsqueeze(2)
    with torch.no_grad():
        if wrapper.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=wrapper.compute_dtype):
                outs = encoder(x)
        else:
            outs = encoder(x)
    return {int(layer): outs[k] for k, layer in enumerate(encoder.hierarchical_layers)}


def _run_once(
    wrapper: VJEPA21Wrapper,
    batch_chw: torch.Tensor,
    *,
    fp32: bool,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    """One equivalence pass. Returns (per-later-tap errors, negative-control errors)."""
    encoder = wrapper.encoder
    tap = _first_tap(encoder)
    # In fp32 mode force encoder weights + autocast off so the comparison is a
    # clean correctness proof, not a bf16 rounding measurement.
    cast = torch.float32 if fp32 else None
    prev_dtype = wrapper.compute_dtype
    if fp32:
        wrapper.compute_dtype = torch.float32

    try:
        with _as_dtype(encoder, cast):
            ref = _reference_hierarchical(wrapper, batch_chw)
            l3_raw = capture_raw_l3(wrapper, batch_chw)

            # geometry the tail needs (image mode, T=1, native 24x78 grid)
            lay = wrapper.layout(num_frames=1)
            tail = Tail(encoder)
            with torch.no_grad():
                if wrapper.device.type == "cuda" and not fp32:
                    with torch.autocast(device_type="cuda", dtype=wrapper.compute_dtype):
                        recon = tail(l3_raw, T=1, H_patches=lay.grid_h, W_patches=lay.grid_w)
                        # negative control: feed the *normed* L3 instead of raw
                        neg = tail(ref[tap], T=1, H_patches=lay.grid_h, W_patches=lay.grid_w)
                else:
                    recon = tail(l3_raw, T=1, H_patches=lay.grid_h, W_patches=lay.grid_w)
                    neg = tail(ref[tap], T=1, H_patches=lay.grid_h, W_patches=lay.grid_w)
    finally:
        wrapper.compute_dtype = prev_dtype

    later = [b for b in encoder.hierarchical_layers if b > tap]
    errs = {b: _errors(recon[b], ref[b]) for b in later}
    neg_final = encoder.hierarchical_layers[-1]
    neg_err = _errors(neg[neg_final], ref[neg_final])
    return errs, neg_err


def run_equivalence(
    sequence_nr: int = 0,
    n_frames: int = 3,
    fp32_atol: float = 1e-4,
) -> bool:
    """End-to-end check on real KITTI frames. Returns True iff the fp32 gate passes."""
    from jepa_drive_wm.data.kitti import KITTISequence

    wrapper = VJEPA21Wrapper(size=VJEPA21Size.BASE)
    seq = KITTISequence(sequence_nr)
    frames = seq.left_images[:n_frames]
    batch = wrapper._stack_frames(frames)  # (B,C,H,W), same preprocessing as the probe
    lay = wrapper.layout(num_frames=1)
    expected_n = lay.grid_h * lay.grid_w
    print(f"[l3-equiv] seq {sequence_nr:02d}, {len(frames)} frames, grid "
          f"{lay.grid_h}x{lay.grid_w}={expected_n} tokens, device {wrapper.device}")

    # ---- token-order / shape sanity ----
    l3 = capture_raw_l3(wrapper, batch)
    B, N, D = l3.shape
    assert (B, N, D) == (len(frames), expected_n, wrapper.size.embed_dim), \
        f"unexpected raw-L3 shape {(B, N, D)}"
    print(f"[l3-equiv] raw-L3 shape OK: {(B, N, D)}")

    ok = True

    # ---- fp32: the correctness gate ----
    print("\n[l3-equiv] === fp32 (hard gate, autocast off) ===")
    errs, neg = _run_once(wrapper, batch, fp32=True)
    for b, e in sorted(errs.items()):
        passed = e["max_abs"] < fp32_atol
        ok = ok and passed
        flag = "PASS" if passed else "FAIL"
        print(f"  block {b:2d}: max_abs={e['max_abs']:.2e}  mean_abs={e['mean_abs']:.2e}  "
              f"cos_min={e['cos_min']:.6f}  [{flag} < {fp32_atol:.0e}]")
    # negative control: normed-L3 fed to the tail must NOT reproduce the reference
    neg_breaks = neg["max_abs"] > fp32_atol * 10
    print(f"  neg-control (normed-L3 -> tail): max_abs={neg['max_abs']:.2e} "
          f"[{'discriminates OK' if neg_breaks else 'WARNING: did not break'}]")
    ok = ok and neg_breaks

    # ---- bf16: informational error floor (only meaningful on cuda) ----
    if wrapper.device.type == "cuda":
        print("\n[l3-equiv] === bf16 (production dtype, informational) ===")
        errs_bf, _ = _run_once(wrapper, batch, fp32=False)
        for b, e in sorted(errs_bf.items()):
            print(f"  block {b:2d}: max_abs={e['max_abs']:.2e}  mean_abs={e['mean_abs']:.2e}  "
                  f"cos_min={e['cos_min']:.6f}")
    else:
        print("\n[l3-equiv] (cpu: skipping bf16 pass — compute dtype is already fp32)")

    print(f"\n[l3-equiv] RESULT: {'PASS ✅ architecture valid' if ok else 'FAIL ❌'}")
    return ok


if __name__ == "__main__":
    import sys
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    raise SystemExit(0 if run_equivalence(sequence_nr=seq) else 1)
