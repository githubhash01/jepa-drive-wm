# Projected Predictor

## Geometry-guided latent forecasting with V-JEPA 2.1 on KITTI

This document specifies a geometry-guided latent forecasting system for KITTI. The central idea is to use deterministic 3D forward projection wherever the future image is geometrically observable, and to use the pretrained V-JEPA 2.1 predictor to supply a learned spatiotemporal latent that corrects motion errors and completes information that geometry cannot provide.

The desired output is not RGB. It is a sequence of future **static image-mode V-JEPA 2.1 ViT-B feature maps** that remain compatible with existing frozen depth and semantic decoders.

The proposed system is therefore divided into three components:

1. a deterministic RGB-D forward projection module;
2. the pretrained V-JEPA 2.1 ViT-B video encoder and predictor, intercepted at its internal 384-dimensional representation;
3. a trainable residual latent update model that combines geometry, the previous static state, and the 384-dimensional predictor state.

The exact architecture of the residual update model is intentionally left as a design TODO. Its mathematical role and required properties are specified precisely.

---

## 1. Nomenclature

### 1.1 Time indexing

Frames are indexed at the chosen forecasting interval. In the current KITTI experiments, adjacent model timesteps are separated by

$$
\Delta t = 0.5\ \mathrm{s}.
$$

Thus, if the latest observed frame is indexed by $t$, then

$$
I_{t+1} = I(t+0.5\ \mathrm{s}),
\qquad
I_{t+2} = I(t+1.0\ \mathrm{s}).
$$

The proposed learned rollout is limited initially to two future static states:

$$
\hat Z_{t+1},\qquad \hat Z_{t+2}.
$$

### 1.2 Image and patch dimensions

The working KITTI image resolution is

$$
H=384,
\qquad
W=1248,
\qquad
C_{\mathrm{RGB}}=3.
$$

The V-JEPA patch size is

$$
P=16.
$$

Therefore, the static spatial token grid is

$$
H_p = \frac{H}{P}=24,
\qquad
W_p = \frac{W}{P}=78,
$$

with

$$
S=H_pW_p=24\cdot78=1872
$$

spatial tokens per image.

The V-JEPA video tokenizer uses a temporal tubelet size

$$
\tau=2.
$$

For a clip containing $T$ RGB frames, the video token grid therefore has

$$
T_p=\frac{T}{2}
$$

temporal planes and

$$
N=T_pH_pW_p
$$

total video tokens.

### 1.3 Core symbols

| Symbol | Shape at KITTI resolution | Meaning |
|---|---:|---|
| $I_i$ | $H\times W\times3$ | Real RGB image at timestep $i$. |
| $D_i$ | $H\times W$ | Metric camera $z$-depth for $I_i$. |
| $K$ | $3\times3$ | Camera intrinsic matrix. |
| $T_{b\leftarrow a}$ | $4\times4$ | Rigid transform mapping coordinates from frame $a$ into frame $b$. |
| $\bar I_{i\rightarrow j}$ | $H\times W\times3$ | RGB image obtained by forward-projecting $I_i$ into camera $j$. |
| $Q_{i\rightarrow j}^{\mathrm{pix}}$ | $H\times W$ | Binary pixel occupancy produced by the forward splat. |
| $Q_{i\rightarrow j}$ | $H_p\times W_p$ | Fractional patch coverage in $[0,1]$. |
| $S_{i\rightarrow j}$ | $H_p\times W_p$ | Binary valid-patch mask after thresholding coverage. |
| $C_{i\rightarrow j}$ | set of indices | Geometrically retained or context patches. |
| $M_{i\rightarrow j}$ | set of indices | Missing or target patches, with $M=\Omega\setminus C$. |
| $E_B^{\mathrm{img}}$ | image $\mapsto S\times768$ | Frozen V-JEPA 2.1 ViT-B encoder in image mode. |
| $E_B^{\mathrm{vid}}$ | video $\mapsto N\times768$ | V-JEPA 2.1 ViT-B encoder in video mode. |
| $Z_i^\star$ | $H_p\times W_p\times768$ | Frozen image-mode ViT-B latent of the real frame $I_i$. This is the static target state. |
| $Z_{i\rightarrow j}^{\mathrm{warp}}$ | $H_p\times W_p\times768$ | Image-mode ViT-B latent of the projected RGB image $\bar I_{i\rightarrow j}$. |
| $\widetilde Z_j$ | $H_p\times W_p\times768$ | Complete provisional future state formed from warp latents and carried-forward static latents. |
| $V_i$ | $H_p\times W_p\times384$ | Proposed semantic transport/completion latent obtained from the internal V-JEPA predictor state for transition $i\rightarrow i+1$. |
| $F_\theta$ | model | Trainable residual latent update model. Exact architecture is TODO. |
| $\Delta Z_{i+1}$ | $H_p\times W_p\times768$ | Residual correction predicted by $F_\theta$. |
| $\hat Z_{i+1}$ | $H_p\times W_p\times768$ | Final predicted static image-mode latent. |

### 1.4 Important terminology distinction

The released V-JEPA 2.1 ViT-B predictor was distilled to predict the final-layer feature space of a frozen ViT-G teacher. Its standard output head maps

$$
384\longrightarrow1664.
$$

The proposed system does **not** treat this 1664-dimensional output as its final prediction. Instead, it intercepts the predictor immediately before that projection and treats the normalized 384-dimensional hidden grid as a trainable spatiotemporal transport/completion representation:

$$
V_i\in\mathbb{R}^{H_p\times W_p\times384}.
$$

The desired future state remains the 768-dimensional static image-mode ViT-B representation:

$$
\hat Z_{i+1}\in\mathbb{R}^{H_p\times W_p\times768}.
$$

---

## 2. Problem formulation

At the latest observed time $t$, assume that the system has:

- the RGB image $I_t$;
- a sufficiently accurate metric depth map $D_t$;
- camera intrinsics $K$;
- proposed future ego motions $T_{t+k\leftarrow t}$ for $k\in\{1,2\}$, or for a longer set of horizons during data generation;
- a history of observed RGB frames used by the V-JEPA video pathway.

A canonical four-frame context is

$$
\mathcal I_{\mathrm{ctx}}
=
\left\{
I_{t-3},I_{t-2},I_{t-1},I_t
\right\},
$$

corresponding, at $\Delta t=0.5\ \mathrm{s}$, to observations at

$$
-1.5,\ -1.0,\ -0.5,\ 0.0\ \mathrm{s}
$$

relative to the latest observation. The exact video clip arrangement used to obtain consecutive predictor transport fields is discussed in Section 26.

The deterministic geometry module produces partial future RGB observations

$$
\bar I_{t\rightarrow t+k}
=
\mathcal W\!\left(I_t,D_t,K,T_{t+k\leftarrow t}\right).
$$

As the horizon increases, the fraction of the future field of view that can be explained from $I_t$ decreases because of:

- points leaving the image;
- newly visible surfaces;
- depth discontinuities;
- limited point-splat support;
- violations of the static-world assumption, especially independently moving vehicles.

The system therefore does not ask a learned model to predict the whole future from scratch. Instead, it decomposes the task into

$$
\text{future state}
=
\text{geometric proposal}
+
\text{learned correction and completion}.
$$

The final latent forecast is

$$
\hat Z_{i+1}
=
\widetilde Z_{i+1}
+
F_\theta\!\left(
Z_i,
\widetilde Z_{i+1},
V_i,
Q_{i\rightarrow i+1}
\right).
$$

The central hypothesis is that the pretrained V-JEPA predictor body already contains useful temporal structure, and that end-to-end fine-tuning can reorganize its 384-dimensional hidden state into a semantic transport/completion field suitable for correcting future-aligned static latents.

---

# Part I — Deterministic RGB forward projection

## 3. Coordinate conventions

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

## 4. Pinhole projection and unprojection

### 4.1 Forward projection

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

### 4.2 RGB-D unprojection

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

## 5. Ego motion and rigid camera transformation

### 5.1 Ego-motion parameterization

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

### 5.2 Future camera pose in source coordinates

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

### 5.3 Transforming static points into the future camera

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

### 5.4 Source-relative future motions

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

## 6. Projection into the future image

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

## 7. Point splatting and visibility

### 7.1 Why splatting is required

The projected coordinates $(u_j,v_j)$ are generally non-integer. A forward renderer must decide how each point contributes to neighbouring pixels.

The current implementation uses `PyTorch3D` point rasterization with a finite point radius. The point radius in normalized device coordinates is

$$
r_{\mathrm{NDC}}
=
\frac{2r_{\mathrm{px}}}{\min(H,W)}.
$$

This conversion follows the PyTorch3D convention that the shorter image dimension spans two NDC units.

### 7.2 Hard nearest-depth rendering

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

### 7.3 Pixel occupancy

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

## 8. Patch coverage and geometric validity

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

The current code uses a configurable threshold, for example

$$
\tau_{\mathrm{cov}}=0.7.
$$

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

## 9. Geometric assumptions and failure modes

### 9.1 Correctly modelled regions

The warp is strongest when:

- depth is accurate;
- the scene point is static in the world;
- the point remains in the future field of view;
- the target view is not disoccluded;
- the splat density is sufficient;
- calibration and ego motion are accurate.

In these regions, experiments with PCA visualization and frozen depth/semantic decoders indicate that the image-mode latent of the warped RGB is often close to the latent of the true future image.

This empirical result motivates preserving the warped latent as the baseline of the learned system rather than replacing it with a fully learned prediction.

### 9.2 Disocclusion

A surface hidden at time $i$ but visible at $i+1$ has no source point in $\mathcal P_i$. No deterministic source-to-target warp can recover its RGB appearance. These locations naturally fall into the missing set when coverage is insufficient.

### 9.3 Finite splat support

Even a visible static surface can contain small holes because the source sampling grid and target sampling grid do not align. Point radius controls the trade-off between holes and excessive blurring/overlap.

### 9.4 Dynamic objects

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

# Part II — V-JEPA 2.1 encoder and predictor

## 10. Image-mode and video-mode tokenization

V-JEPA 2.1 has modality-specific tokenizers feeding a shared transformer encoder.

### 10.1 Image mode

A single image is represented as a tensor with temporal length one:

$$
I_i\in\mathbb R^{B\times3\times1\times H\times W}.
$$

The image tokenizer is implemented as a 3D convolution with temporal kernel one, which is equivalent to a learned $16\times16$ image patch projection:

$$
\operatorname{kernel}
=
\operatorname{stride}
=
(1,16,16).
$$

For ViT-B, every image patch becomes a 768-dimensional token:

$$
X_i^{\mathrm{img},0}
\in
\mathbb R^{B\times S\times768}.
$$

At KITTI resolution,

$$
X_i^{\mathrm{img},0}
\in
\mathbb R^{B\times1872\times768}.
$$

An image modality embedding is added, and the token sequence passes through the shared 12-block ViT-B encoder. Normal inference returns the normalized final layer:

$$
Z_i^\star
=
E_B^{\mathrm{img}}(I_i)
\in
\mathbb R^{B\times1872\times768}.
$$

Reshaping gives

$$
Z_i^\star
\in
\mathbb R^{B\times24\times78\times768}.
$$

### 10.2 Video mode

A video clip is patchified with

$$
\operatorname{kernel}
=
\operatorname{stride}
=
(2,16,16).
$$

Each initial token is a learned projection of a non-overlapping spatiotemporal cuboid containing

$$
2\times16\times16\times3=1536
$$

RGB values.

For ViT-B, every tubelet becomes a 768-dimensional token:

$$
X^{\mathrm{vid},0}
\in
\mathbb R^{B\times N\times768},
$$

where

$$
N=\frac{T}{2}H_pW_p.
$$

For eight KITTI frames,

$$
T_p=4,
$$

$$
N=4\cdot24\cdot78=7488.
$$

The video modality embedding is added, and the tokens pass through the same ViT-B transformer weights used by the image pathway.

The initial token has a literal $2\times16\times16$ receptive field. After self-attention, however, each token is a globally contextualized representation anchored at that tubelet coordinate rather than a purely local cuboid descriptor.

---

## 11. Where masking occurs

The context encoder does not receive blacked-out mask tokens inside its transformer.

The sequence is:

1. run the learned image/video patch embedding;
2. add the appropriate modality embedding;
3. use `apply_masks` to gather only the retained token indices;
4. pass only those context tokens into the encoder transformer.

If the full patch sequence is

$$
X^0
=
\begin{bmatrix}
x_0\\x_1\\\vdots\\x_{N-1}
\end{bmatrix},
$$

and the context index set is

$$
C=\{c_1,\ldots,c_{N_C}\},
$$

then the live encoder receives

$$
X_C^0
=
\begin{bmatrix}
x_{c_1}\\\vdots\\x_{c_{N_C}}
\end{bmatrix}.
$$

The target tokens are absent. There are no zero rows and no learned mask tokens in the encoder sequence.

This is essential because otherwise context tokens could absorb information from the target content through encoder self-attention before masking.

---

## 12. Original V-JEPA 2.1 pretraining versus released ViT-B distillation

Two training regimes must be kept separate.

### 12.1 Full V-JEPA 2.1 pretraining

For the full ViT-g/ViT-G pretraining recipe, V-JEPA 2.1 uses deep self-supervision. Several intermediate encoder outputs are normalized and concatenated. The predictor consumes this multi-level representation and predicts several teacher levels.

For a 12-layer ViT-B-shaped example, the selected levels would be

$$
[2,5,8,11],
$$

and a four-level 768-dimensional concatenation would have dimension

$$
4\cdot768=3072.
$$

This is the source of the 3072-dimensional hierarchical representation discussed during the architectural analysis.

### 12.2 Released distilled ViT-B checkpoint

The released V-JEPA 2.1 ViT-B checkpoint is different. It is distilled from a frozen ViT-G teacher.

During distillation:

- the context/student encoder is ViT-B;
- the frozen target teacher is ViT-G;
- only the final encoder layer is used;
- deep self-supervision is disabled;
- the predictor has 12 transformer blocks;
- the predictor embedding dimension is 384;
- the final prediction head matches the 1664-dimensional ViT-G teacher space;
- an EMA copy of the ViT-B student is maintained separately and becomes the released downstream encoder.

Thus the released ViT-B predictor pathway is

$$
768
\longrightarrow
384
\longrightarrow
\underbrace{384\longrightarrow\cdots\longrightarrow384}_{12\ \mathrm{predictor\ blocks}}
\longrightarrow
1664.
$$

The proposed projected predictor uses this released distilled model, not the four-level 3072-dimensional pretraining pathway.

---

## 13. Predictor input construction

Let the masked ViT-B video encoder output final-layer context features

$$
Z_C^{B}
\in
\mathbb R^{B\times N_C\times768}.
$$

### 13.1 Input projection

Because only one encoder level is used for the distilled ViT-B predictor, `predictor_embed` is a single learned linear map:

$$
U_C
=
Z_C^B W_{\mathrm{in}}+\mathbf b_{\mathrm{in}},
$$

with

$$
W_{\mathrm{in}}
\in
\mathbb R^{768\times384}.
$$

Therefore,

$$
U_C
\in
\mathbb R^{B\times N_C\times384}.
$$

### 13.2 Learned target placeholders

The predictor contains a bank of learned mask-token parameters

$$
\mathbf m_k
\in
\mathbb R^{384}.
$$

A selected mask token is copied into every target position. If the target set is

$$
M=\{m_1,\ldots,m_{N_M}\},
$$

then initially

$$
U_M^0
=
\begin{bmatrix}
\mathbf m\\\vdots\\\mathbf m
\end{bmatrix}
\in
\mathbb R^{N_M\times384}.
$$

The checkpoint mask token is not a mean patch embedding and is not a direct future prediction. It is a learned generic state representing unknown content.

All target positions begin with the same selected content vector. Their different spatiotemporal roles are supplied by their token indices through RoPE.

### 13.3 Restoring spatiotemporal order

The predictor initially holds context and target tokens in two groups:

$$
[U_C;U_M^0].
$$

It concatenates their integer position IDs, sorts by those IDs, and applies the same permutation to the token sequence. This reconstructs the original sparse/full spatiotemporal ordering before predictor self-attention.

---

## 14. Spatiotemporal position IDs and 3D RoPE

For a video token at temporal plane $\tau$, patch row $h$, and patch column $w$, define the flattened global token ID

$$
j(\tau,h,w)
=
\tau H_pW_p+hW_p+w.
$$

The inverse mapping is

$$
\tau
=
\left\lfloor
\frac{j}{H_pW_p}
\right\rfloor,
$$

$$
h
=
\left\lfloor
\frac{j\bmod(H_pW_p)}{W_p}
\right\rfloor,
$$

$$
w
=
j\bmod W_p.
$$

The predictor passes these integer IDs to each attention block. Its 3D RoPE implementation divides part of every attention head into temporal, height, and width subspaces and rotates the query and key channels according to $(\tau,h,w)$.

The released ViT-B predictor uses 12 attention heads. Since its hidden dimension is 384, each head has

$$
d_h=\frac{384}{12}=32
$$

channels. In the current RoPE implementation, 10 channels are assigned to time, 10 to height, and 10 to width; the remaining two channels are left unrotated.

For one attention head,

$$
q_i=W_Qx_i,
\qquad
k_i=W_Kx_i,
\qquad
v_i=W_Vx_i.
$$

RoPE produces

$$
\widetilde q_i
=
R(\tau_i,h_i,w_i)q_i,
$$

$$
\widetilde k_i
=
R(\tau_i,h_i,w_i)k_i.
$$

The value vector is not rotated.

The attention score between tokens $i$ and $j$ is

$$
s_{ij}
=
\frac{
\widetilde q_i^\top\widetilde k_j
}{\sqrt{d_h}},
$$

and the attention weights are

$$
a_{ij}
=
\frac{\exp(s_{ij})}{\sum_{r}\exp(s_{ir})}.
$$

The output for token $i$ is

$$
o_i
=
\sum_j a_{ij}v_j.
$$

This is the point where a learned target placeholder first receives content-dependent information from context tokens.

Because the predictor is non-causal in the released masked-prediction setup, context and target tokens can all communicate. Target tokens can also communicate with other target tokens.

---

## 15. One predictor transformer block

Let

$$
X^{(\ell)}
\in
\mathbb R^{B\times N\times384}
$$

be the input to predictor block $\ell$.

The block is pre-normalized.

### 15.1 Attention sublayer

First,

$$
\bar X^{(\ell)}
=
\operatorname{LN}_1\!\left(X^{(\ell)}\right).
$$

Multi-head RoPE self-attention produces

$$
A^{(\ell)}
=
\operatorname{MHSA}_{\mathrm{RoPE}}
\left(
\bar X^{(\ell)}
\right).
$$

The residual update is

$$
X_{\mathrm{attn}}^{(\ell)}
=
X^{(\ell)}+A^{(\ell)}.
$$

### 15.2 MLP sublayer

Then,

$$
\widetilde X_{\mathrm{attn}}^{(\ell)}
=
\operatorname{LN}_2
\left(
X_{\mathrm{attn}}^{(\ell)}
\right),
$$

$$
G^{(\ell)}
=
\operatorname{MLP}
\left(
\widetilde X_{\mathrm{attn}}^{(\ell)}
\right),
$$

and

$$
X^{(\ell+1)}
=
X_{\mathrm{attn}}^{(\ell)}+G^{(\ell)}.
$$

For the ViT-B predictor,

$$
384
\longrightarrow
1536
\longrightarrow
384
$$

is the usual MLP width when `mlp_ratio = 4`.

The sequence dimension and channel dimension remain unchanged across all 12 blocks:

$$
X^{(0)},\ldots,X^{(12)}
\in
\mathbb R^{B\times N\times384}.
$$

---

## 16. Standard predictor output and the proposed interception point

After the 12 predictor blocks, the code applies

$$
H
=
\operatorname{LN}_{\mathrm{pred}}
\left(
X^{(12)}
\right),
$$

where

$$
H
\in
\mathbb R^{B\times N\times384}.
$$

The standard distilled model then projects each token into the ViT-G teacher space:

$$
Y^G
=
HW_G+\mathbf b_G,
$$

with

$$
W_G
\in
\mathbb R^{384\times1664}.
$$

The proposed system intercepts the computation before $W_G$.

The old 1664-dimensional head is therefore not required in the main forecast path:

$$
\boxed{
V
:=
H
\in
\mathbb R^{B\times N\times384}
}
$$

is treated as the learned spatiotemporal transport/completion field.

### 16.1 Context and target hidden states

The released predictor can return both target and context outputs through separate projection heads. For the proposed system, the useful quantity is instead the **unprojected 384-dimensional state for both subsets**.

After undoing the sorting permutation, let

$$
H_C\in\mathbb R^{B\times N_C\times384}
$$

and

$$
H_M\in\mathbb R^{B\times N_M\times384}.
$$

Scatter both subsets back into the complete grid:

$$
V(p)
=
\begin{cases}
H_C(p),&p\in C,\\
H_M(p),&p\in M.
\end{cases}
$$

This full field is important. Dynamic-object errors can occur in geometrically valid context patches, so the residual model must receive predictor states at both valid and masked positions.

A source-level modification should expose these hidden tensors immediately after `predictor_norm` and before `predictor_proj` or `predictor_proj_context`.

### 16.2 Original dense target and context supervision

V-JEPA 2.1 does not supervise only the learned mask-token positions. Its dense predictive objective also predicts visible context tokens. In the full pretraining recipe, the masked-token prediction receives an $L_1$ loss, while nearby context-token predictions receive a distance-weighted $L_1$ loss. The distilled ViT-B recipe retains the same dense objective but applies it only to the final frozen ViT-G teacher layer.

Conceptually, with teacher targets $Y^\star$, the two terms are

$$
\mathcal L_{\mathrm{target}}
=
\frac{1}{|M|}
\sum_{p\in M}
\left\|
\hat Y(p)-Y^\star(p)
\right\|_1,
$$

and

$$
\mathcal L_{\mathrm{context}}
=
\frac{1}{\sum_{p\in C}w_p}
\sum_{p\in C}
w_p
\left\|
\hat Y(p)-Y^\star(p)
\right\|_1,
$$

where $w_p$ is larger for context locations considered relevant to the masked region under the V-JEPA masking geometry.

This source architecture is useful for the proposed system for two reasons:

1. the predictor already updates and predicts context tokens rather than treating them as immutable memory;
2. the proposed valid-region loss is a natural adaptation of the same principle, but its purpose is explicitly to clean geometrically supplied future content, including dense dynamic-object errors.

---

## 17. Non-square KITTI RoPE requirement

The KITTI token grid is

$$
24\times78,
$$

not square.

The V-JEPA RoPE attention implementation supports explicit values of `H_patches` and `W_patches`. When those values are not supplied, it decodes flattened token IDs under a square-grid fallback:

$$
\text{tokens per frame}=\texttt{grid\_size}^2.
$$

In the supplied predictor, each block is currently called as

```python
x, attn = blk(x, mask=masks)
```

without the true grid height and width. At KITTI resolution, this would decode global token IDs incorrectly.

The predictor must pass the actual grid dimensions:

```python
x, attn = blk(
    x,
    mask=masks,
    T=current_grid_depth,
    H_patches=24,
    W_patches=78,
)
```

More generally,

```python
H_patches = image_height // patch_size
W_patches = image_width  // patch_size
```

must be propagated into every predictor block.

This change does not alter any learned parameter shape. It corrects only the interpretation of position IDs.

---

# Part III — Proposed projected predictor

## 18. High-level system

The proposed system combines four sources of information for each transition $i\rightarrow i+1$:

1. the current static image state $Z_i$;
2. a future-aligned geometric proposal $Z_{i\rightarrow i+1}^{\mathrm{warp}}$;
3. a binary or continuous warp-validity field $S_{i\rightarrow i+1}$ or $Q_{i\rightarrow i+1}$;
4. a learned 384-dimensional V-JEPA predictor field $V_i$.

The prediction is

$$
\hat Z_{i+1}
=
\widetilde Z_{i+1}
+
\Delta Z_{i+1},
$$

where

$$
\Delta Z_{i+1}
=
F_\theta
\left(
Z_i,
\widetilde Z_{i+1},
V_i,
Q_{i\rightarrow i+1}
\right).
$$

The residual model is not asked to reconstruct the scene from nothing. It starts from a full provisional future latent and modifies it only where required.

---

## 19. Producing the warped future latent

The forward projector produces

$$
\bar I_{i\rightarrow i+1}^{\mathrm{masked}}
$$

and the patch-validity map

$$
S_{i\rightarrow i+1}.
$$

The masked warped RGB is passed through the frozen V-JEPA 2.1 ViT-B image encoder:

$$
Z_{i\rightarrow i+1}^{\mathrm{warp}}
=
E_B^{\mathrm{img}}
\left(
\bar I_{i\rightarrow i+1}^{\mathrm{masked}}
\right).
$$

Thus,

$$
Z_{i\rightarrow i+1}^{\mathrm{warp}}
\in
\mathbb R^{24\times78\times768}.
$$

Although invalid RGB patches are zeroed, their image-mode tokens should not be trusted. They are replaced in the next step.

A subtle implementation detail is that the full image transformer sees the zeroed patches during image-mode encoding, so valid token features are still contextualized in the presence of those holes. The current empirical tests indicate that the resulting valid-region features remain useful. A future ablation may compare this with alternative RGB or latent infill before image encoding.

---

## 20. Provisional complete future state

Define the binary patch-validity map

$$
S_{i+1}(p)
\in
\{0,1\}.
$$

For valid geometric patches, use the warped future latent. For invalid patches, copy the previous static latent at the same patch coordinate:

$$
\widetilde Z_{i+1}(p)
=
S_{i+1}(p)
Z_{i\rightarrow i+1}^{\mathrm{warp}}(p)
+
\left(1-S_{i+1}(p)\right)
Z_i(p).
$$

Equivalently,

$$
\widetilde Z_{i+1}
=
S_{i+1}\odot Z_{i\rightarrow i+1}^{\mathrm{warp}}
+
(1-S_{i+1})\odot Z_i.
$$

This has a simple interpretation:

- geometry provides the initial future state wherever it has sufficient support;
- missing patches begin from a zero-motion semantic prior.

The copied $Z_i$ patches are not assumed to be correctly positioned at $i+1$. They merely prevent the residual model from starting from empty or arbitrary vectors.

---

## 21. Conservative masking for video tubelets

The V-JEPA video tokenizer consumes two RGB frames per temporal token plane. A tubelet at spatial location $(h,w)$ is computed jointly from both constituent image patches.

Therefore, if a future video pair is

$$
\left(
\bar I_i,
\bar I_{i+1}
\right),
$$

a retained video tubelet should ideally contain reliable RGB support in both frames.

A conservative shared mask is

$$
S_i^{\mathrm{tube}}(h,w)
=
S_i(h,w)\land S_{i+1}(h,w).
$$

Using continuous coverage, an equivalent rule is

$$
Q_i^{\mathrm{tube}}(h,w)
=
\min
\left(
Q_i(h,w),Q_{i+1}(h,w)
\right),
$$

$$
S_i^{\mathrm{tube}}(h,w)
=
\mathbf 1
\left[
Q_i^{\mathrm{tube}}(h,w)
\ge\tau_{\mathrm{cov}}
\right].
$$

This discards some usable near-frame pixels but prevents a retained 3D tubelet from mixing reliable RGB in one frame with a largely invalid patch in the other.

The mask is applied after the video patch embedding and before the ViT-B context transformer. Missing tubelets are later reintroduced as learned 384-dimensional predictor mask tokens.

---

## 22. The 384-dimensional semantic transport latent

Let

$$
V_i
\in
\mathbb R^{24\times78\times384}
$$

denote the complete predictor hidden field associated with the transition $i\rightarrow i+1$.

It is not assumed that the pretrained checkpoint already represents optical flow or transport explicitly. Initially, it is simply the internal representation that was useful for predicting masked ViT-G video features.

However, after end-to-end fine-tuning under the proposed loss, it is expected to encode whatever information best supports the residual update, potentially including:

- ego-motion-consistent semantic displacement;
- independent object motion;
- temporal correspondence;
- object persistence;
- occlusion and disocclusion cues;
- uncertainty about warped context;
- appearance or semantic change;
- completion information for genuinely missing content.

Thus the word **transport** is an intended functional role, not a claim that $V_i$ is a literal optical-flow field.

The predictor is initialized from the released V-JEPA 2.1 weights. Therefore, the model is fine-tuned under a new objective rather than trained from random initialization.

---

## 23. Residual latent update

The residual model receives

$$
Z_i
\in
\mathbb R^{24\times78\times768},
$$

$$
\widetilde Z_{i+1}
\in
\mathbb R^{24\times78\times768},
$$

$$
V_i
\in
\mathbb R^{24\times78\times384},
$$

and

$$
Q_{i\rightarrow i+1}
\in
\mathbb R^{24\times78\times1}.
$$

It predicts

$$
\Delta Z_{i+1}
=
F_\theta
\left(
Z_i,
\widetilde Z_{i+1},
V_i,
Q_{i\rightarrow i+1}
\right),
$$

with

$$
\Delta Z_{i+1}
\in
\mathbb R^{24\times78\times768}.
$$

The final prediction is

$$
\boxed{
\hat Z_{i+1}
=
\widetilde Z_{i+1}
+
\Delta Z_{i+1}
}.
$$

### 23.1 Required properties of $F_\theta$

The exact architecture is TODO, but it must satisfy the following requirements.

#### Spatial mixing

A purely patchwise MLP cannot move a vehicle representation from one patch to another. The model must permit information exchange between spatial locations through at least one of:

- spatial self-attention;
- cross-attention from future locations into $Z_i$;
- deformable sampling;
- convolutional residual blocks with sufficient receptive field;
- another explicit transport operator.

Conceptually, it must be capable of operations resembling

$$
Z_{i+1}^{\mathrm{transport}}(q)
=
\sum_p A_{qp}(V_i)Z_i(p),
$$

where $A_{qp}$ describes how semantic content at current location $p$ contributes to future location $q$.

#### Corrections on valid patches

The residual must be allowed to modify all locations:

$$
\Delta Z_{i+1}(p)
\neq0
$$

is permitted even when

$$
S_{i+1}(p)=1.
$$

This is necessary for dynamic objects and other dense-but-wrong splat errors.

#### Coverage as an input, not a hard gate

Do not enforce

$$
\Delta Z_{i+1}
=
(1-Q_{i+1})F_\theta(\cdot).
$$

Such a gate would suppress corrections exactly where a moving vehicle has high splat coverage but incorrect motion.

Instead, $Q$ is evidence that the network may learn to trust or override depending on temporal and semantic context.

#### Residual initialization

The final output layer of $F_\theta$ should be zero-initialized so that at optimization step zero

$$
\Delta Z_{i+1}=0,
$$

and therefore

$$
\hat Z_{i+1}=\widetilde Z_{i+1}.
$$

Training begins from the strong geometry-plus-copy-forward baseline already validated qualitatively.

---

## 24. Correcting dynamic objects inside valid splat regions

Suppose a car occupies patch $p$ at timestep $i$. Static-scene geometry projects it to patch $q$, but its true independent motion places it at patch $r$.

The warped latent may contain

$$
Z_{i\rightarrow i+1}^{\mathrm{warp}}(q)
\approx
\text{car},
$$

while the true future latent contains

$$
Z_{i+1}^\star(q)
\approx
\text{background},
$$

and

$$
Z_{i+1}^\star(r)
\approx
\text{car}.
$$

The model must therefore learn both

$$
\text{remove car semantics at }q
$$

and

$$
\text{insert or transport car semantics to }r.
$$

This is feasible only if:

1. $F_\theta$ has spatial mixing;
2. $V_i$ is available at valid and masked positions;
3. valid warped patches receive training loss;
4. the geometric validity mask is not used as an immutable gate.

The context-region loss is therefore not merely a cleanup auxiliary. It is the supervision that teaches the system to override geometrically dense but dynamically incorrect content.

---

## 25. Two-step daisy-chained rollout

The initial system predicts only two future static latents.

### 25.1 First step

The known current state is

$$
Z_t
=
Z_t^\star
$$

during the first rollout step.

Construct

$$
\widetilde Z_{t+1}
=
S_{t+1}\odot Z_{t\rightarrow t+1}^{\mathrm{warp}}
+
(1-S_{t+1})\odot Z_t.
$$

Then

$$
\hat Z_{t+1}
=
\widetilde Z_{t+1}
+
F_\theta
\left(
Z_t,
\widetilde Z_{t+1},
V_t,
Q_{t+1}
\right).
$$

### 25.2 Second step

The second step uses the autoregressively predicted first state:

$$
\widetilde Z_{t+2}
=
S_{t+2}\odot Z_{t\rightarrow t+2}^{\mathrm{warp}}
+
(1-S_{t+2})\odot \hat Z_{t+1}.
$$

Then

$$
\hat Z_{t+2}
=
\widetilde Z_{t+2}
+
F_\theta
\left(
\hat Z_{t+1},
\widetilde Z_{t+2},
V_{t+1},
Q_{t+2}
\right).
$$

The same residual model parameters $\theta$ are shared across both transitions.

Because the second loss backpropagates through $\hat Z_{t+1}$, the first prediction is trained not only to match the first future target but also to remain a useful state for subsequent rollout.

---

## 26. Consecutive transition latents and temporal staggering

The standard V-JEPA video tokenizer uses non-overlapping temporal tubelets with stride two. A clip

$$
[I_0,I_1,I_2,I_3,\ldots]
$$

naturally produces transition-aligned planes for

$$
(I_0,I_1),
\quad
(I_2,I_3),
\quad\ldots
$$

It does not simultaneously produce an overlapping plane for $(I_1,I_2)$.

If the rollout requires consecutive transport fields

$$
V_t
\text{ for }t\rightarrow t+1
$$

and

$$
V_{t+1}
\text{ for }t+1\rightarrow t+2,
$$

one clean solution is to run two temporally staggered video/predictor passes:

- an even-aligned pass producing one subset of adjacent transitions;
- a one-frame-shifted pass producing the complementary subset.

The resulting transport fields are interleaved before the two-step static rollout.

The exact clip construction and reuse of shared context are implementation details to finalize, but the stride-two alignment constraint must be respected.

---

# Part IV — Training objective

## 27. Frozen target space

The desired output space is defined by the released ViT-B encoder in image mode:

$$
Z_i^\star
=
E_{B,\mathrm{frozen}}^{\mathrm{img}}(I_i).
$$

This encoder must remain frozen because:

1. it defines the regression target;
2. the existing depth and semantic decoders were trained on this feature space;
3. moving the encoder would move the target coordinate system and invalidate decoder compatibility.

The warped image latent is computed with the same frozen encoder:

$$
Z_{t\rightarrow t+k}^{\mathrm{warp}}
=
E_{B,\mathrm{frozen}}^{\mathrm{img}}
\left(
\bar I_{t\rightarrow t+k}^{\mathrm{masked}}
\right).
$$

If the forecasting video encoder is fine-tuned, it should be a separate model copy initialized from the same released EMA checkpoint. The frozen image target encoder must remain separate.

---

## 28. Trainable components

The intended trainable forecasting path is

$$
E_B^{\mathrm{vid}}
\rightarrow
\text{predictor input projection}
\rightarrow
\text{12 predictor blocks}
\rightarrow
V_i^{384}
\rightarrow
F_\theta
\rightarrow
\hat Z_{i+1}^{768}.
$$

The system is initialized from pretrained V-JEPA 2.1 ViT-B encoder and predictor weights.

Possible optimization stages are:

1. train $F_\theta$ while freezing the V-JEPA video encoder and predictor;
2. fine-tune the predictor with a smaller learning rate;
3. optionally fine-tune the video encoder with an even smaller learning rate.

This staged schedule is not an architectural requirement. Fully end-to-end training of the forecasting path is possible, provided that the frozen target image encoder remains separate.

The pretrained predictor is not equivalent to a random initialization. The input projection, mask-token bank, RoPE attention weights, MLPs, LayerNorms, and all 12 predictor blocks begin from a model trained to infer missing video representations from context.

---

## 29. Patchwise latent distance

For a predicted and target patch latent

$$
\hat{\mathbf z},\mathbf z^\star\in\mathbb R^{768},
$$

a useful base distance is

$$
d(\hat{\mathbf z},\mathbf z^\star)
=
\alpha
\left\|
\hat{\mathbf z}-\mathbf z^\star
\right\|_1
+
\beta
\left(
1-
\frac{
\hat{\mathbf z}^{\top}\mathbf z^\star
}{
\|\hat{\mathbf z}\|_2
\|\mathbf z^\star\|_2+\varepsilon
}
\right).
$$

The exact choice between $L_1$, Smooth-$L_1$, $L_2$, cosine distance, or a combination remains an experimental decision.

---

## 30. Separate valid-context and missing-region losses

A single average over all patches can hide poor performance in missing regions if most of the image is already well explained by the warp.

For future step $k$, define

$$
\mathcal L_{k}^{\mathrm{valid}}
=
\frac{1}{|C_{t+k}|}
\sum_{p\in C_{t+k}}
 d
\left(
\hat Z_{t+k}(p),
Z_{t+k}^\star(p)
\right),
$$

and

$$
\mathcal L_{k}^{\mathrm{missing}}
=
\frac{1}{|M_{t+k}|}
\sum_{p\in M_{t+k}}
 d
\left(
\hat Z_{t+k}(p),
Z_{t+k}^\star(p)
\right).
$$

The per-step loss is

$$
\mathcal L_k
=
\lambda_C
\mathcal L_k^{\mathrm{valid}}
+
\lambda_M
\mathcal L_k^{\mathrm{missing}}.
$$

The two-step objective is

$$
\boxed{
\mathcal L_{\mathrm{latent}}
=
\lambda_1\mathcal L_1
+
\lambda_2\mathcal L_2
}.
$$

The valid-region loss teaches geometric cleanup and dynamic-object correction. The missing-region loss teaches completion and transport into unobserved areas.

---

## 31. Optional dense-decoder consistency

The ultimate quality criterion is not only latent similarity but dense decodability.

Let

$$
D_{\mathrm{depth}}
$$

and

$$
D_{\mathrm{sem}}
$$

be the existing frozen depth and semantic decoders trained on static image-mode ViT-B features.

Optional auxiliary losses are

$$
\mathcal L_{\mathrm{depth}}
=
\ell_{\mathrm{depth}}
\left(
D_{\mathrm{depth}}(\hat Z_{t+k}),
D_{\mathrm{depth}}(Z_{t+k}^\star)
\right),
$$

and

$$
\mathcal L_{\mathrm{sem}}
=
\ell_{\mathrm{sem}}
\left(
D_{\mathrm{sem}}(\hat Z_{t+k}),
D_{\mathrm{sem}}(Z_{t+k}^\star)
\right).
$$

These can be interpreted as decoder-space distillation from the true future latent. They should be introduced only after establishing the behaviour of the direct latent objective.

The total loss may later become

$$
\mathcal L
=
\mathcal L_{\mathrm{latent}}
+
\gamma_D\mathcal L_{\mathrm{depth}}
+
\gamma_S\mathcal L_{\mathrm{sem}}.
$$

---

## 32. Recommended evaluation decomposition

Performance should be reported separately on:

1. valid static-background patches;
2. valid dynamic-object patches;
3. missing or low-coverage patches;
4. all future patches;
5. rollout step $t+1$;
6. rollout step $t+2$.

This separation reveals whether the system is:

- preserving good geometry;
- correcting dense-but-wrong dynamic splats;
- filling true holes;
- remaining stable under autoregressive rollout.

Useful comparisons include:

$$
Z_{t\rightarrow t+k}^{\mathrm{warp}}
$$

versus

$$
\widetilde Z_{t+k}
$$

versus

$$
\hat Z_{t+k}
$$

versus

$$
Z_{t+k}^\star.
$$

Depth and semantic outputs should be decoded from each latent baseline using the same frozen decoders.

---

# Part V — End-to-end computation

## 33. Training-time forward pass

For a training sample with current state $t$ and two future steps:

### 33.1 Geometry

Compute

$$
\left(
\bar I_{t\rightarrow t+1},
Q_{t+1},
S_{t+1}
\right)
=
\mathcal W
\left(
I_t,D_t,K,T_{t+1\leftarrow t}
\right),
$$

$$
\left(
\bar I_{t\rightarrow t+2},
Q_{t+2},
S_{t+2}
\right)
=
\mathcal W
\left(
I_t,D_t,K,T_{t+2\leftarrow t}
\right).
$$

### 33.2 Frozen static latents

Compute

$$
Z_t^\star
=
E_B^{\mathrm{img}}(I_t),
$$

$$
Z_{t+1}^\star
=
E_B^{\mathrm{img}}(I_{t+1}),
$$

$$
Z_{t+2}^\star
=
E_B^{\mathrm{img}}(I_{t+2}),
$$

and

$$
Z_{t\rightarrow t+1}^{\mathrm{warp}}
=
E_B^{\mathrm{img}}(\bar I_{t\rightarrow t+1}),
$$

$$
Z_{t\rightarrow t+2}^{\mathrm{warp}}
=
E_B^{\mathrm{img}}(\bar I_{t\rightarrow t+2}).
$$

### 33.3 Video predictor states

Construct the observed-plus-warped video input, apply conservative tubelet masks, run the ViT-B video context encoder, insert learned predictor mask tokens, and obtain complete 384-dimensional fields

$$
V_t,
\qquad
V_{t+1}.
$$

If required by the stride-two tubelet alignment, obtain these fields from staggered predictor passes.

### 33.4 First static prediction

Form

$$
\widetilde Z_{t+1}
=
S_{t+1}\odot Z_{t\rightarrow t+1}^{\mathrm{warp}}
+
(1-S_{t+1})\odot Z_t^\star,
$$

then

$$
\hat Z_{t+1}
=
\widetilde Z_{t+1}
+
F_\theta
\left(
Z_t^\star,
\widetilde Z_{t+1},
V_t,
Q_{t+1}
\right).
$$

### 33.5 Second static prediction

Form

$$
\widetilde Z_{t+2}
=
S_{t+2}\odot Z_{t\rightarrow t+2}^{\mathrm{warp}}
+
(1-S_{t+2})\odot\hat Z_{t+1},
$$

then

$$
\hat Z_{t+2}
=
\widetilde Z_{t+2}
+
F_\theta
\left(
\hat Z_{t+1},
\widetilde Z_{t+2},
V_{t+1},
Q_{t+2}
\right).
$$

### 33.6 Loss

Compute

$$
\mathcal L
=
\lambda_1
\mathcal L
\left(
\hat Z_{t+1},Z_{t+1}^\star
\right)
+
\lambda_2
\mathcal L
\left(
\hat Z_{t+2},Z_{t+2}^\star
\right),
$$

with valid and missing regions normalized separately.

Backpropagation runs through both residual updates and through the trainable V-JEPA forecasting path according to the selected fine-tuning schedule.

---

## 34. Inference-time forward pass

At inference, future RGB frames and future target latents are unavailable.

The system receives:

- observed RGB history;
- latest RGB $I_t$;
- latest depth $D_t$;
- intrinsics $K$;
- future ego-motion hypotheses.

It then:

1. forward-warps $I_t$ to future horizons;
2. computes patch coverage and valid/missing sets;
3. encodes the warped RGB images into image-mode ViT-B latents;
4. runs the video context encoder and predictor to obtain $V_t$ and $V_{t+1}$;
5. constructs $\widetilde Z_{t+1}$ and predicts $\hat Z_{t+1}$;
6. constructs $\widetilde Z_{t+2}$ using $\hat Z_{t+1}$ in missing regions;
7. predicts $\hat Z_{t+2}$;
8. decodes depth and semantics from the two predicted static latents if required.

The output is

$$
\left(
\hat Z_{t+1},
\hat Z_{t+2}
\right)
\in
\mathbb R^{2\times24\times78\times768}.
$$

---

# Part VI — Implementation requirements and open decisions

## 35. Required source changes

### 35.1 Expose the predictor hidden state

Modify `predictor.forward` so that it can return the normalized 384-dimensional context and target states before the 1664-dimensional output heads.

Conceptually:

```python
x = self.predictor_norm(x)
x = undo_sort(x, reverse_argsort)

h_context = x[:, :N_ctxt, :]   # [B, Nc, 384]
h_target  = x[:, N_ctxt:, :]   # [B, Nm, 384]

return h_target, h_context
```

Then scatter both subsets into a complete spatiotemporal field using `masks_x` and `masks_y`.

### 35.2 Pass true non-square dimensions into RoPE

Every predictor block must receive

```python
H_patches = 24
W_patches = 78
```

for the current KITTI resolution.

### 35.3 Preserve a frozen target encoder

If the video encoder is fine-tuned, instantiate a separate frozen ViT-B image encoder for:

- $Z_i^\star$ targets;
- warped static latents;
- compatibility with frozen dense decoders.

### 35.4 Preserve mask provenance

The residual model must receive at least:

- binary patch validity $S$;
- preferably continuous coverage $Q$.

The system must be able to distinguish a token copied from $Z_i$ from a token obtained from warped RGB.

---

## 36. Residual architecture TODO

The exact form of $F_\theta$ remains unresolved.

The first implementation should be the smallest architecture satisfying the spatial-mixing requirement. Candidate families include:

- a shallow spatial transformer over the $24\times78$ grid;
- windowed cross-attention from future locations into $Z_i$;
- deformable latent sampling guided by $V_i$;
- a compact ConvNeXt-style residual network;
- a hybrid transport-plus-residual block.

The architecture should not duplicate the full capacity of the V-JEPA predictor. The predictor already performs the expensive joint spatiotemporal reasoning. $F_\theta$ should primarily translate that information into corrections of the future-aligned 768-dimensional static state.

---

## 37. Outstanding experimental questions

1. How much useful transition information exists in the frozen 384-dimensional predictor state before fine-tuning?
2. Is fine-tuning only the predictor sufficient, or must the video context encoder also adapt?
3. Which residual spatial operator best corrects moving vehicles without damaging static regions?
4. Should continuous patch coverage replace the binary validity decision in the provisional-state blend?
5. Does RGB or latent infill before image-mode encoding reduce contamination from black invalid patches?
6. What is the best conservative mask rule for a two-frame tubelet?
7. How should the two temporally staggered predictor passes share computation?
8. How much does second-step autoregressive training improve rollout stability?
9. Should the old 384-to-1664 ViT-G head be retained as an auxiliary regularizer during early fine-tuning?
10. Which latent loss best preserves performance under the frozen depth and semantic decoders?

---

## 38. Summary

The proposed system is built around the following decomposition:

$$
\boxed{
\text{future static latent}
=
\text{geometry-aligned proposal}
+
\text{learned semantic correction/completion}
}
$$

Geometry converts metric depth, intrinsics, ego motion, and the current RGB frame into partial future RGB observations. The image-mode ViT-B encoder converts these observations into strong future-aligned static latent proposals.

The V-JEPA 2.1 video encoder and predictor process observed and partially warped video context. Rather than using the predictor's original 1664-dimensional teacher-space output, the system intercepts the 384-dimensional hidden field and fine-tunes it to function as a semantic transport/completion representation.

Missing future latent patches begin from the corresponding current static state, while valid future patches begin from the warped static latent. A trainable spatial residual model then corrects both subsets:

$$
\hat Z_{i+1}
=
\left[
S_{i+1}\odot Z_{i\rightarrow i+1}^{\mathrm{warp}}
+
(1-S_{i+1})\odot Z_i
\right]
+
F_\theta
\left(
Z_i,
\widetilde Z_{i+1},
V_i,
Q_{i+1}
\right).
$$

This formulation preserves the information that deterministic projection already gets right, while allowing the learned model to:

- fill disocclusions and geometric holes;
- transport semantics across patch locations;
- correct moving vehicles even inside high-coverage splat regions;
- clean splat artefacts;
- produce static 768-dimensional feature maps compatible with existing dense decoders.

The first model predicts only two future static states and is trained through the complete two-step autoregressive rollout.

---

## 39. Source correspondence

The mathematical geometry section corresponds to the supplied `warp_rgb.py` implementation, particularly:

- `ego_motion_to_camera_pose`;
- `ego_motion_to_warp_se3`;
- `_rgbd_to_pointcloud`;
- `PointRasterizer.forward`;
- `_make_patch_mask`;
- `warp_sequence`.

The encoder and predictor sections correspond to the supplied V-JEPA 2.1 source files and the current official implementation:

- `app/vjepa_2_1/models/vision_transformer.py`;
- `app/vjepa_2_1/models/predictor.py`;
- `app/vjepa_2_1/models/utils/modules.py`;
- `src/hub/backbones.py`;
- *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*.

Official references:

- [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482)
- [Official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
- [ViT-B/ViT-L model construction](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py)
- [V-JEPA 2.1 predictor](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/predictor.py)
- [V-JEPA 2.1 encoder](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py)
- [Transformer and 3D RoPE modules](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/utils/modules.py)