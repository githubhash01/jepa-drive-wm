# Shared nomenclature

This file defines the notation used consistently by:

- [Deterministic RGB-D forward projection](./warp_rgb.md)
- [V-JEPA 2.1 encoder and predictor](./vjepa21_architecture.md)
- [Projected predictor](./projected_predictor.md)

## Fixed dimensions

The working KITTI resolution is

$$
H=384,\qquad W=1248,\qquad P=16,
$$

so the image-token grid is

$$
H_p=H/P=24,\qquad W_p=W/P=78,\qquad S=H_pW_p=1872.
$$

The V-JEPA video tubelet size is

$$
\tau=2.
$$

A clip with $T$ RGB frames therefore produces

$$
T_p=T/2,\qquad N=T_pH_pW_p
$$

video tokens. The chosen model timestep is normally

$$
\Delta t=0.5\ \mathrm{s}.
$$

## Coordinate and transform conventions

Camera axes follow KITTI/OpenCV convention:

$$
+x=\text{right},\qquad +y=\text{down},\qquad +z=\text{forward}.
$$

$T_{b\leftarrow a}\in SE(3)$ maps a homogeneous point from camera frame $a$ into camera frame $b$:

$$
\widetilde{\mathbf X}_b=T_{b\leftarrow a}\widetilde{\mathbf X}_a.
$$

## Core symbols

| Symbol | Shape | Meaning |
|---|---:|---|
| $I_i$ | $H\times W\times3$ | Real RGB frame at model timestep $i$. |
| $D_i$ | $H\times W$ | Metric camera $z$-depth for $I_i$. |
| $K$ | $3\times3$ | Camera intrinsic matrix at the working resolution. |
| $\bar I_{i\rightarrow j}$ | $H\times W\times3$ | RGB forward projection of $I_i$ into camera $j$. |
| $Q^{\mathrm{pix}}_{i\rightarrow j}$ | $H\times W$ | Binary rasterizer occupancy. |
| $Q_{i\rightarrow j}$ | $H_p\times W_p$ | Fraction of occupied pixels in each patch. |
| $S_{i\rightarrow j}$ | $H_p\times W_p$ | Binary patch-validity mask after thresholding $Q$. |
| $C_{i\rightarrow j}$ | set | Retained/context patch indices. |
| $M_{i\rightarrow j}$ | set | Missing/target patch indices, $M=\Omega\setminus C$. |
| $E_B^{\mathrm{img}}$ | image $\mapsto S\times768$ | Frozen released V-JEPA 2.1 ViT-B encoder in image mode. |
| $E_B^{\mathrm{vid}}$ | video $\mapsto N\times768$ | V-JEPA 2.1 ViT-B encoder in video mode. |
| $Z_i^\star$ | $H_p\times W_p\times768$ | Frozen image-mode ViT-B latent of the real frame $I_i$. |
| $Z_i$ | $H_p\times W_p\times768$ | State supplied to a rollout step: $Z_i^\star$ under teacher forcing or $\hat Z_i$ autoregressively. |
| $Z_{i\rightarrow j}^{\mathrm{warp}}$ | $H_p\times W_p\times768$ | Image-mode ViT-B encoding of $\bar I_{i\rightarrow j}$. |
| $Z_j^0$ | $H_p\times W_p\times768$ | Deterministic warp-plus-copy future proposal before learned correction. |
| $V_i$ | $H_p\times W_p\times384$ | Unprojected V-JEPA predictor hidden field for transition $i\rightarrow i+1$; proposed transport/completion latent. |
| $\Pi_\Delta$ | $384\rightarrow768$ | Shared learned projection from predictor hidden state to static-latent correction. |
| $\Delta\hat Z_{i+1}$ | $H_p\times W_p\times768$ | Learned correction relative to the deterministic proposal, $\Pi_\Delta(V_i)$. |
| $\hat Z_{i+1}$ | $H_p\times W_p\times768$ | Predicted future static image-mode latent. |

Unless stated otherwise, batch dimensions are omitted for readability. Flattened token index $j(\tau,h,w)$ follows temporal-major order:

$$
j(\tau,h,w)=\tau H_pW_p+hW_p+w.
$$
