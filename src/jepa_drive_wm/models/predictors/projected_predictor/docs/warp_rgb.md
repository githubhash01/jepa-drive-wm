# Deterministic RGB-D forward projection

> Shared notation: [nomenclature.md](./nomenclature.md)  
> Companion documents: [V-JEPA 2.1 architecture](./vjepa21_architecture.md) · [Projected predictor](./projected_predictor.md)

This document explains the supplied `warp_rgb.py` module as a geometric operation. The module reconstructs a coloured point cloud from one RGB-D observation, transforms the source camera to a proposed future pose, renders the static point cloud from that pose with PyTorch3D, and converts rasterizer occupancy into a conservative $16\times16$ patch-validity mask.

It deliberately performs **no learned inpainting and no dynamic-object modelling**. Its output is a partial future observation and an explicit statement of where geometric evidence exists.

## Computation at a glance

$$
(I_i,D_i,K,T_{j\leftarrow i})
\xrightarrow{\text{unproject}}
\mathcal P_i
\xrightarrow{T_{j\leftarrow i}}
\mathcal P_j
\xrightarrow{\text{project and splat}}
(\bar I_{i\rightarrow j},Q^{\mathrm{pix}}_{i\rightarrow j})
\xrightarrow{\text{patch pooling}}
(Q_{i\rightarrow j},S_{i\rightarrow j}).
$$

The mathematical derivation below follows the implementation in these functions:

- `ego_motion_to_camera_pose` and `ego_motion_to_warp_se3`;
- `_rgbd_to_pointcloud`;
- `_target_camera` and `cameras_from_opencv_projection`;
- `PointRasterizer.forward`;
- `_make_patch_mask`;
- `warp_sequence`.

---

## Coordinate conventions

The implementation treats the source camera at time $i$ as the world frame. It uses the KITTI/OpenCV camera convention

$$
+x=\text{right},
\qquad
+y=\text{down},
\qquad
+z=\text{forward}.
$$

A source-camera point is written

$$
\mathbf X_i
=
\begin{bmatrix}
X_i\\Y_i\\Z_i
\end{bmatrix}
\in\mathbb R^3.
$$

A pixel location is written

$$
\mathbf p
=
\begin{bmatrix}
u\\v
\end{bmatrix},
\qquad
\widetilde{\mathbf p}
=
\begin{bmatrix}
u\\v\\1
\end{bmatrix}.
$$

The intrinsic matrix is

$$
K
=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}.
$$

When an image is resized from $(H_0,W_0)$ to $(H,W)$, the intrinsics must be scaled consistently. Let

$$
s_x=\frac{W}{W_0},
\qquad
s_y=\frac{H}{H_0}.
$$

Then

$$
K'
=
\begin{bmatrix}
s_x&0&0\\
0&s_y&0\\
0&0&1
\end{bmatrix}K,
$$

so that $f_x$ and $c_x$ scale by $s_x$, while $f_y$ and $c_y$ scale by $s_y$. This is the operation performed in the current KITTI example before constructing the warp camera. RGB is resized bilinearly and metric depth is resized with nearest-neighbour sampling.

The depth map stores metric camera $z$-depth:

$$
d(u,v)=Z_i.
$$

This is not Euclidean range along the viewing ray. It is the third coordinate in the camera frame.

---

## Pinhole projection and unprojection

### Forward projection

For a point with positive depth $Z_i>0$, the pinhole camera equations are

$$
u
=
f_x\frac{X_i}{Z_i}+c_x,
$$

$$
v
=
f_y\frac{Y_i}{Z_i}+c_y.
$$

Equivalently,

$$
\lambda\widetilde{\mathbf p}
=
K\mathbf X_i,
$$

where $\lambda=Z_i$ under the chosen depth convention.

### RGB-D unprojection

Given a valid source pixel $(u,v)$ and its depth $d(u,v)$, the corresponding 3D point in the source camera frame is

$$
\mathbf X_i(u,v)
=
d(u,v)K^{-1}\widetilde{\mathbf p}.
$$

Expanded component-wise,

$$
X_i
=
\frac{u-c_x}{f_x}d(u,v),
$$

$$
Y_i
=
\frac{v-c_y}{f_y}d(u,v),
$$

$$
Z_i
=
d(u,v).
$$

Each valid pixel therefore creates a coloured 3D point

$$
\left(\mathbf X_i(u,v),\ I_i(u,v)\right).
$$

The implementation keeps the physical camera model in the OpenCV/KITTI convention. Internally, `cameras_from_opencv_projection` constructs the equivalent PyTorch3D camera, and `get_screen_to_ndc_transform(..., with_xyflip=True)` converts image coordinates $(+x$ right, $+y$ down$)$ into PyTorch3D NDC coordinates $(+x$ left, $+y$ up$)$. PyTorch3D then unprojects using metric view-space $z$. This coordinate conversion changes the library representation, not the underlying pinhole equations above.

The complete source RGB-D observation becomes a coloured point cloud

$$
\mathcal P_i
=
\left\{
\left(\mathbf X_i^n,\mathbf c_i^n\right)
\right\}_{n=1}^{N_{\mathrm{valid}}}.
$$

In `warp_rgb.py`, invalid points are rejected if depth is non-finite, below `min_depth`, or paired with non-finite RGB.

---

## Ego motion and rigid camera transformation

### Ego-motion parameterization

The current warp implementation receives

$$
\mathbf e
=
\begin{bmatrix}
f\\r\\\psi
\end{bmatrix},
$$

where

- $f$ is forward camera translation in metres;
- $r$ is rightward camera translation in metres;
- $\psi$ is rightward yaw in radians.

The future camera centre expressed in the source frame is

$$
\mathbf C_i
=
\begin{bmatrix}
r\\0\\f
\end{bmatrix}.
$$

Positive yaw is a positive rotation about the camera $+y$ axis, rotating the forward $+z$ direction toward $+x$:

$$
R_y(\psi)
=
\begin{bmatrix}
\cos\psi & 0 & \sin\psi\\
0 & 1 & 0\\
-\sin\psi & 0 & \cos\psi
\end{bmatrix}.
$$

### Future camera pose in source coordinates

The implementation constructs

$$
T_{i\leftarrow j}
=
\begin{bmatrix}
R_{i\leftarrow j} & \mathbf C_i\\
\mathbf 0^\top & 1
\end{bmatrix},
$$

where $j$ denotes the future camera and

$$
\mathbf X_i
=
R_{i\leftarrow j}\mathbf X_j+\mathbf C_i.
$$

This transform describes the future camera pose in source coordinates.

### Transforming static points into the future camera

To render the static source point cloud from the future camera, the transform is inverted:

$$
T_{j\leftarrow i}
=
T_{i\leftarrow j}^{-1}
=
\begin{bmatrix}
R_{i\leftarrow j}^{\top}
&
-R_{i\leftarrow j}^{\top}\mathbf C_i\\
\mathbf 0^\top&1
\end{bmatrix}.
$$

Thus, for every static world point,

$$
\mathbf X_j
=
R_{i\leftarrow j}^{\top}
\left(
\mathbf X_i-\mathbf C_i
\right).
$$

This is the geometric core of the forward warp. The 3D points remain fixed in the source/world frame while the camera moves.

### Source-relative future motions

Every future motion supplied to `warp_sequence` is relative to the same source frame:

$$
T_{t+1\leftarrow t},
\quad
T_{t+2\leftarrow t},
\quad
\ldots
$$

They are not assumed to be incremental transforms of the form

$$
T_{t+2\leftarrow t+1}.
$$

The source point cloud is built once and rendered from each proposed future camera pose.

---

## Projection into the future image

For a transformed point

$$
\mathbf X_j
=
\begin{bmatrix}
X_j\\Y_j\\Z_j
\end{bmatrix},
\qquad
Z_j>0,
$$

the future pixel is

$$
u_j
=
f_x\frac{X_j}{Z_j}+c_x,
$$

$$
v_j
=
f_y\frac{Y_j}{Z_j}+c_y.
$$

Conceptually, the complete geometric mapping is

$$
\widetilde{\mathbf p}_j
\sim
K
R_{i\leftarrow j}^{\top}
\left(
 d_i(u_i,v_i)K^{-1}\widetilde{\mathbf p}_i
-
\mathbf C_i
\right).
$$

Because the operation starts from source pixels and sends them to future positions, this is a **forward warp**. Multiple source points can land near the same target pixel, while some target pixels receive no source point at all.

---

## Point splatting and visibility

### Why splatting is required

The projected coordinates $(u_j,v_j)$ are generally non-integer. A forward renderer must decide how each point contributes to neighbouring pixels.

The current implementation uses `PyTorch3D` point rasterization with a finite point radius. The point radius in normalized device coordinates is

$$
r_{\mathrm{NDC}}
=
\frac{2r_{\mathrm{px}}}{\min(H,W)}.
$$

This conversion follows the PyTorch3D convention that the shorter image dimension spans two NDC units.

### Hard nearest-depth rendering

The rasterizer is configured with

```python
points_per_pixel = 1
```

so only the nearest point at each covered target pixel is retained. If several source points splat onto the same future pixel, the selected point is

$$
n^\star(u,v)
=
\arg\min_{n\in\mathcal N(u,v)} Z_j^n.
$$

The rendered colour is

$$
\bar I_{i\rightarrow j}(u,v)
=
\mathbf c_i^{n^\star(u,v)}.
$$

The corresponding rendered depth is

$$
\bar D_{i\rightarrow j}(u,v)
=
Z_j^{n^\star(u,v)}.
$$

This is a hard $z$-buffer. It handles static-scene visibility between projected points but does not infer content for target pixels receiving no point.

### Pixel occupancy

The raw geometric occupancy mask is

$$
Q_{i\rightarrow j}^{\mathrm{pix}}(u,v)
=
\begin{cases}
1,&\text{if at least one point covers }(u,v),\\
0,&\text{otherwise}.
\end{cases}
$$

Importantly, occupancy means only that geometry supplied a point. It does not mean that the projected content is semantically correct.

---

## Patch coverage and geometric validity

For each V-JEPA patch $(a,b)$, define its pixel support

$$
\mathcal R_{a,b}
=
\left\{
(u,v):
16a\le v<16(a+1),
\quad
16b\le u<16(b+1)
\right\}.
$$

The fractional geometric coverage is

$$
Q_{i\rightarrow j}(a,b)
=
\frac{1}{P^2}
\sum_{(u,v)\in\mathcal R_{a,b}}
Q_{i\rightarrow j}^{\mathrm{pix}}(u,v).
$$

A patch is retained when

$$
S_{i\rightarrow j}(a,b)
=
\mathbf 1
\left[
Q_{i\rightarrow j}(a,b)
\ge
\tau_{\mathrm{cov}}
\right].
$$

The threshold is defined once as `DEFAULT_PATCH_COVERAGE_THRESHOLD` in `warp_rgb.py` and is currently

$$
\tau_{\mathrm{cov}}=0.8.
$$

Empirically (seq-9 sweep, 2026-08-20), a higher threshold makes $S$ a much better trust signal: a "valid" patch still carries up to $(1-\tau_{\mathrm{cov}})$ unoccupied black pixels inside its own $16\times16$ token, corrupting it before any attention. At $0.7$ the warp latent barely beat copy-forward on valid patches; $0.8$ roughly quadrupled that margin, while $0.9$ starved the far horizon of geometric evidence.

The context and missing sets are

$$
C_{i\rightarrow j}
=
\left\{
(a,b):S_{i\rightarrow j}(a,b)=1
\right\},
$$

$$
M_{i\rightarrow j}
=
\left\{
(a,b):S_{i\rightarrow j}(a,b)=0
\right\}.
$$

For visualization and image-mode encoding, the current implementation zeros all pixels in invalid patches:

$$
\bar I_{i\rightarrow j}^{\mathrm{masked}}(u,v)=0
\quad
\text{if }(u,v)\text{ lies in an invalid patch}.
$$

No RGB hole filling is performed in the current deterministic warper.

---

## Geometric assumptions and failure modes

### Correctly modelled regions

The warp is strongest when:

- depth is accurate;
- the scene point is static in the world;
- the point remains in the future field of view;
- the target view is not disoccluded;
- the splat density is sufficient;
- calibration and ego motion are accurate.

In these regions, experiments with PCA visualization and frozen depth/semantic decoders indicate that the image-mode latent of the warped RGB is often close to the latent of the true future image.

This empirical result motivates preserving the warped latent as the baseline of the learned system rather than replacing it with a fully learned prediction.

### Disocclusion

A surface hidden at time $i$ but visible at $i+1$ has no source point in $\mathcal P_i$. No deterministic source-to-target warp can recover its RGB appearance. These locations naturally fall into the missing set when coverage is insufficient.

### Finite splat support

Even a visible static surface can contain small holes because the source sampling grid and target sampling grid do not align. Point radius controls the trade-off between holes and excessive blurring/overlap.

### Dynamic objects

The most important limitation is that the transform assumes a static world. A moving vehicle has object motion in addition to camera ego motion:

$$
T_{\mathrm{object}}
\neq I.
$$

The correct future point should depend on both camera and object motion. The current warp applies only the camera transform. Therefore a moving car may be projected to an incorrect future location even when the patch has high occupancy:

$$
Q_{i\rightarrow i+1}(a,b)\approx1,
$$

while

$$
Z_{i\rightarrow i+1}^{\mathrm{warp}}(a,b)
\not\approx
Z_{i+1}^{\star}(a,b).
$$

This is a **dense-but-wrong** failure, not a missing-data failure. Consequently:

> Patch coverage must be treated as a confidence feature, not as a hard statement that the warped content is correct.

The learned residual must be allowed to modify both valid and invalid geometric regions.

---
## Code correspondence

| Mathematical operation | `warp_rgb.py` implementation |
|---|---|
| Scale calibration to $384\times1248$ | example block updating the first two rows of $K$ |
| Source RGB-D unprojection | `_rgbd_to_pointcloud` |
| Future camera pose | `ego_motion_to_camera_pose` |
| World-to-future-camera transform | `ego_motion_to_warp_se3` and `_invert_se3` |
| OpenCV-to-PyTorch3D camera conversion | `_camera_from_opencv_extrinsics` |
| Target-view point rendering | `PointRasterizer.forward` |
| Hard visibility selection | `points_per_pixel=1` and `fragments.zbuf` |
| Patch coverage and validity | `_make_patch_mask` |
| Reuse one source point cloud for several horizons | `warp_sequence` |

## Precedent and scope

The module sits in the classical geometry-first lineage of view synthesis:

1. **Pinhole reprojection and view synthesis.** Depth-and-pose systems such as Zhou et al. use calibrated camera geometry to relate pixels across viewpoints. Their common differentiable implementation is inverse sampling; the present module instead uses source-to-target forward projection.
2. **Forward splatting.** Forward warping naturally creates collisions and holes. Softmax Splatting studies differentiable collision resolution for video interpolation. The present implementation uses a simpler hard nearest-depth rule because the immediate purpose is deterministic evidence construction, not learned image synthesis.
3. **Point-cloud novel-view synthesis.** SynSin unprojects source features to a point cloud, renders them from a target camera, and learns to refine missing regions. FWD similarly uses explicit depth and forward warping for fast novel-view synthesis. The proposed project follows the same geometry-then-completion decomposition, but delegates completion to a latent V-JEPA predictor rather than an RGB decoder.
4. **PyTorch3D.** The implementation uses PyTorch3D's camera conversion and point rasterizer rather than a custom projection kernel.

The important boundary is that these references justify the geometric operator and its failure modes; they do not imply that the current hard splat is photorealistic or dynamically correct.

## References

1. T. Zhou, M. Brown, N. Snavely, and D. G. Lowe, **Unsupervised Learning of Depth and Ego-Motion from Video**, CVPR 2017. <https://arxiv.org/abs/1704.07813>
2. O. Wiles, G. Gkioxari, R. Szeliski, and J. Johnson, **SynSin: End-to-end View Synthesis from a Single Image**, CVPR 2020. <https://arxiv.org/abs/1912.08804>
3. A. Cao, C. Rockwell, and J. Johnson, **FWD: Real-time Novel View Synthesis with Forward Warping and Depth**, CVPR 2022. <https://arxiv.org/abs/2206.08355>
4. S. Niklaus and F. Liu, **Softmax Splatting for Video Frame Interpolation**, CVPR 2020. <https://arxiv.org/abs/2003.05534>
5. N. Ravi et al., **Accelerating 3D Deep Learning with PyTorch3D**, 2020. <https://arxiv.org/abs/2007.08501>
6. PyTorch3D, **OpenCV camera conversion documentation**. <https://pytorch3d.readthedocs.io/en/latest/modules/utils.html#pytorch3d.utils.cameras_from_opencv_projection>
7. A. Geiger, P. Lenz, and R. Urtasun, **Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite**, CVPR 2012. <https://www.cvlibs.net/publications/Geiger2012CVPR.pdf>
