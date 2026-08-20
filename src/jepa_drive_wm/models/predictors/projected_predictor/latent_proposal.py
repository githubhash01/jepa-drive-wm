"""Deterministic latent proposal used by the projected predictor.

At each future patch, keep the image-mode latent encoded from the RGB warp when
that patch is geometrically valid; otherwise copy the previous static latent:

    Z^0_{i+1}(p) = Z^warp_{i+1}(p),  p in C
                 = Z_i(p),           p in M

The module has no learnable parameters.  It is used identically for the
zero-learning baseline and inside the autoregressive learned model, where
``z_previous`` may itself be a previously predicted latent.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentProposal(nn.Module):
    """Blend previous and warped V-JEPA image latents using patch validity."""

    def forward(
        self,
        z_previous: torch.Tensor,
        z_warped: torch.Tensor,
        patch_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the deterministic proposal ``Z^0``.

        Parameters
        ----------
        z_previous, z_warped:
            Same-shaped latent tensors with feature dimension last. Supported
            examples include ``(N,D)``, ``(Hp,Wp,D)``, ``(B,N,D)`` and
            ``(B,Hp,Wp,D)``.
        patch_valid:
            Boolean/numeric validity tensor matching every dimension of the
            latents except the final feature dimension, e.g. ``(N,)``,
            ``(Hp,Wp)``, ``(B,N)`` or ``(B,Hp,Wp)``.

        Returns
        -------
        torch.Tensor
            ``z_warped`` on valid patches and ``z_previous`` elsewhere.
        """
        if z_previous.shape != z_warped.shape:
            raise ValueError(
                "z_previous and z_warped must have identical shapes, got "
                f"{tuple(z_previous.shape)} and {tuple(z_warped.shape)}."
            )
        if z_previous.ndim < 2:
            raise ValueError(
                "Latents must have at least one patch/grid dimension and one "
                "feature dimension."
            )
        if z_previous.device != z_warped.device:
            raise ValueError("z_previous and z_warped must be on the same device.")

        expected_mask_shape = z_previous.shape[:-1]
        if tuple(patch_valid.shape) != tuple(expected_mask_shape):
            raise ValueError(
                "patch_valid must match the latent shape excluding the feature "
                f"dimension. Expected {tuple(expected_mask_shape)}, got "
                f"{tuple(patch_valid.shape)}."
            )

        mask = patch_valid.to(device=z_previous.device, dtype=torch.bool).unsqueeze(-1)
        return torch.where(mask, z_warped, z_previous)
