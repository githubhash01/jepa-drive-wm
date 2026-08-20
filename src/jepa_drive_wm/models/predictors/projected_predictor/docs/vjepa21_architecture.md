# V-JEPA 2.1 encoder and predictor

> Shared notation: [nomenclature.md](./nomenclature.md)  
> Companion documents: [RGB-D forward projection](./warp_rgb.md) · [Projected predictor](./projected_predictor.md)

This document traces the computation implemented by the V-JEPA 2.1 encoder and predictor, with particular emphasis on the **released distilled ViT-B/16-384 model** used by this project. It also separates that released checkpoint from the full ViT-g/ViT-G pretraining recipe, because the two objectives use different target representations.

The source files are:

- `app/vjepa_2_1/models/vision_transformer.py`;
- `app/vjepa_2_1/models/predictor.py`;
- `app/vjepa_2_1/models/utils/modules.py`;
- `src/hub/backbones.py`.

## Released ViT-B dimensional summary

$$
\text{ViT-B context feature }768
\xrightarrow{\text{predictor input projection}}
384
\xrightarrow{12\ \text{predictor blocks}}
384
\xrightarrow{\text{distillation head}}
1664.
$$

The final 1664-dimensional space belongs to the frozen ViT-G teacher used during distillation. Normal downstream use discards the predictor and returns the final 768-dimensional representation of the released EMA ViT-B encoder.

---

## Image-mode and video-mode tokenization

V-JEPA 2.1 has modality-specific tokenizers feeding a shared transformer encoder.

### Image mode

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

### Video mode

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

## Where masking occurs

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

## Original V-JEPA 2.1 pretraining versus released ViT-B distillation

Two training regimes must be kept separate.

### Full V-JEPA 2.1 pretraining

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

### Released distilled ViT-B checkpoint

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

The 12-block distilled predictor was trained as a new predictor during distillation; it was not initialized from the 24-block predictor used by the full ViT-G pretraining recipe. The released checkpoint nevertheless contains fully trained predictor weights, which are the initialization used by this project.

---

## Released ViT-B distillation: the three networks

The released ViT-B checkpoint is easiest to understand as three coupled networks during training.

### Frozen ViT-G teacher: complete target path

The frozen ViT-G teacher receives the complete, unmasked image or video sample and returns only its normalized final layer:

$$
Y^\star
=
E_G(x)
\in
\mathbb R^{B\times N\times1664}.
$$

This is the target representation. Intermediate ViT-G layers are computed internally, but they are not concatenated or supervised in the distilled ViT-B objective.

### Live ViT-B student: masked context path

The live ViT-B student patchifies the same sample, removes the target indices before encoder self-attention, and computes final-layer features only for the retained context set:

$$
Z_C^B
=
E_B(x_C)
\in
\mathbb R^{B\times N_C\times768}.
$$

The predictor receives $Z_C^B$, the context addresses $C$, and the target addresses $M$. It reconstructs a complete context-plus-target sequence internally and outputs predictions in the 1664-dimensional teacher space:

$$
(\hat Y_M,\hat Y_C)
=
P(Z_C^B,C,M).
$$

The target and context losses compare these predictions with the corresponding subsets of the unmasked ViT-G teacher representation:

$$
\mathcal L_{\mathrm{target}}
=
\frac{1}{|M|}
\sum_{p\in M}
\left\|
\hat Y_M(p)-Y^\star(p)
\right\|_1,
$$

$$
\mathcal L_{\mathrm{context}}
=
\frac{1}{\sum_{p\in C}w_p}
\sum_{p\in C}
w_p
\left\|
\hat Y_C(p)-Y^\star(p)
\right\|_1.
$$

Thus the predictor learns both to infer masked locations and to maintain localized predictions at visible locations.

### EMA ViT-B student: released inference encoder

A separate EMA copy of the live ViT-B student is updated by Polyak averaging:

$$
\theta_{\mathrm{EMA}}
\leftarrow
\mu\theta_{\mathrm{EMA}}
+
(1-\mu)\theta_{\mathrm{live}}.
$$

This EMA student is not used to construct the distillation loss. It is the model saved and released for downstream use. Consequently, ordinary inference with the public ViT-B checkpoint returns

$$
Z
=
E_{B,\mathrm{EMA}}(x)
\in
\mathbb R^{B\times N\times768},
$$

while the ViT-G teacher and predictor are unnecessary unless one explicitly loads the predictor for a new prediction task.

---

## Predictor input construction

Let the masked ViT-B video encoder output final-layer context features

$$
Z_C^{B}
\in
\mathbb R^{B\times N_C\times768}.
$$

### Input projection

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

### Learned target placeholders

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

### Restoring spatiotemporal order

The predictor initially holds context and target tokens in two groups:

$$
[U_C;U_M^0].
$$

It concatenates their integer position IDs, sorts by those IDs, and applies the same permutation to the token sequence. This reconstructs the original sparse/full spatiotemporal ordering before predictor self-attention.

---

## Spatiotemporal position IDs and 3D RoPE

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

## One predictor transformer block

Let

$$
X^{(\ell)}
\in
\mathbb R^{B\times N\times384}
$$

be the input to predictor block $\ell$.

The block is pre-normalized.

### Attention sublayer

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

### MLP sublayer

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

## Standard predictor output and the proposed interception point

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

### Context and target hidden states

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

### Original dense target and context supervision

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

## Non-square KITTI RoPE requirement

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
## Source-level summary

For the released `vjepa2_1_vit_base_384` constructor, the official Hub source specifies:

- `checkpoint_key="ema_encoder"`;
- `predictor_depth=12`;
- `predictor_num_mask_tokens=8`;
- `n_output_distillation=1`;
- `return_all_tokens=True`;
- `teacher_embed_dim=1664`.

This is why the released ViT-B predictor consumes only the student's final 768-dimensional layer and maps its 384-dimensional hidden states to the 1664-dimensional final layer of the ViT-G teacher.

## References

1. L. Mur-Labadia et al., **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**, 2026. <https://arxiv.org/abs/2603.14482>
2. Meta FAIR, **Official V-JEPA 2 / V-JEPA 2.1 repository**. <https://github.com/facebookresearch/vjepa2>
3. Meta FAIR, **V-JEPA 2.1 encoder source**. <https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py>
4. Meta FAIR, **V-JEPA 2.1 predictor source**. <https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/predictor.py>
5. Meta FAIR, **Transformer and 3D RoPE source**. <https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/utils/modules.py>
6. Meta FAIR, **Released backbone and predictor constructors**. <https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py>
