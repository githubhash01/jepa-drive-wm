# Projected predictor

## Geometry-guided residual prediction in V-JEPA latent space

> Shared notation: [nomenclature.md](./nomenclature.md)  
> Prerequisites: [deterministic RGB-D forward projection](./warp_rgb.md) · [V-JEPA 2.1 encoder and predictor](./vjepa21_architecture.md)

The projected predictor combines a deterministic geometric future proposal with the pretrained V-JEPA 2.1 video predictor.

The first implementation is deliberately simple. No additional spatial residual network is introduced. Instead, the normalized 384-dimensional hidden state of the pretrained V-JEPA predictor is projected directly into a 768-dimensional correction of the static image-mode ViT-B latent.

The complete model is

$$
\boxed{
\hat Z_{i+1}
=
Z_{i+1}^{0}
+
\Delta\hat Z_{i+1}
}
$$

with

$$
\boxed{
\Delta\hat Z_{i+1}
=
\Pi_{\Delta}(V_i),
\qquad
\Pi_{\Delta}:\mathbb R^{384}\rightarrow\mathbb R^{768}.
}
$$

The deterministic term $Z_{i+1}^{0}$ does as much of the prediction as possible using depth, ego motion and RGB forward projection. The learned term corrects whatever remains wrong and fills what geometry cannot observe.

The first system predicts four future static latent states from five observed frames and is trained through the four-step rollout. The four transition fields are obtained with two temporally staggered predictor passes (Section 9).

---

## 1. Static target representation

The desired output space is the frozen V-JEPA 2.1 ViT-B **image-mode** representation.

For a real RGB frame $I_i$,

$$
Z_i^\star
=
E_{B,\mathrm{frozen}}^{\mathrm{img}}(I_i),
$$

where

$$
Z_i^\star
\in
\mathbb R^{24\times78\times768}.
$$

This is the same latent space consumed by the existing frozen depth and semantic decoders.

During rollout,

$$
Z_i
=
\begin{cases}
Z_i^\star, & \text{for an observed state},\\
\hat Z_i, & \text{for a predicted state}.
\end{cases}
$$

The forecasting objective is therefore

$$
\hat Z_{i+1}
\approx
Z_{i+1}^\star.
$$

---

## 2. Deterministic warp-plus-copy proposal

The RGB-D forward projection is defined in [warp_rgb.md](./warp_rgb.md).

From the latest observed RGB-D frame, camera intrinsics and proposed future ego motion, the warper produces

$$
\bar I_{i\rightarrow i+1},
$$

a future RGB projection under the static-world assumption, together with patch coverage

$$
Q_{i+1}\in[0,1]^{24\times78}
$$

and binary patch validity

$$
S_{i+1}\in\{0,1\}^{24\times78}.
$$

Encode the warped RGB in image mode:

$$
Z_{i\rightarrow i+1}^{\mathrm{warp}}
=
E_{B,\mathrm{frozen}}^{\mathrm{img}}
\left(
\bar I_{i\rightarrow i+1}
\right).
$$

Then construct a complete future proposal

$$
\boxed{
Z_{i+1}^{0}
=
S_{i+1}\odot Z_{i\rightarrow i+1}^{\mathrm{warp}}
+
(1-S_{i+1})\odot Z_i.
}
$$

Thus:

- a valid future patch begins from the geometry-aligned warped latent;
- an invalid future patch begins from the previous static latent at the same patch coordinate.

The first case uses a static world assumption, and the second case uses a zero-motion prior (both are wrong). 

With no learned correction,

$$
\hat Z_{i+1}=Z_{i+1}^{0}.
$$

This is the primary zero-learning baseline against which the projected predictor is evaluated.

### 2.1 Geometric validity is not correctness

A high-coverage splat can still be wrong.

The main example is a moving vehicle. The deterministic projector applies ego motion under a static-world assumption, so independently moving objects may be projected densely into the wrong future location.

Therefore,

$$
S_{i+1}(p)=1
$$

does not imply

$$
Z_{i\rightarrow i+1}^{\mathrm{warp}}(p)
\approx
Z_{i+1}^{\star}(p).
$$

The learned model must be able to change both valid and invalid regions.

---

## 3. V-JEPA video predictor pathway

The learned pathway retains the existing V-JEPA 2.1 ViT-B video encoder and predictor as far as possible.

The detailed architecture is described in [vjepa21_architecture.md](./vjepa21_architecture.md). Only the modifications relevant to the projected predictor are repeated here.

### 3.1 Video-mode context

Observed RGB history and geometrically projected future RGB are assembled into a video clip.

The ViT-B video tokenizer uses a learned 3D patch projection with kernel and stride

$$
(2,16,16),
$$

producing 768-dimensional tubelet tokens.

At KITTI resolution, each temporal tubelet plane contains

$$
24\times78=1872
$$

tokens.

Future tubelets judged unreliable are removed after patch embedding and before the ViT-B context transformer, following the normal V-JEPA masking mechanism.

The retained context tokens pass through the pretrained ViT-B encoder, giving

$$
X_C^{\mathrm{vid}}
\in
\mathbb R^{N_C\times768}.
$$

### 3.2 Predictor input

The released distilled ViT-B predictor maps each context feature as

$$
768\rightarrow384.
$$

For target tubelets, it inserts its pretrained learned mask-token vector

$$
m\in\mathbb R^{384}.
$$

The original mask-token mechanism is preserved. The mask token is not replaced with a hand-constructed residual or image latent.

The predictor therefore receives a joint sequence of

$$
\text{projected 384-D context tokens}
+
\text{learned 384-D target placeholders}.
$$

Context and target tokens are ordered by their spatiotemporal patch indices and processed jointly by the pretrained predictor transformer.

### 3.3 Predictor hidden state

After the 12 predictor blocks, V-JEPA applies its final predictor LayerNorm:

$$
V
=
\operatorname{LN}_{\mathrm{pred}}(X).
$$

The normalized hidden state has dimension

$$
384
$$

per video token.

For a transition $i\rightarrow i+1$, denote the relevant future hidden field by

$$
\boxed{
V_i
\in
\mathbb R^{24\times78\times384}.
}
$$

The released predictor would normally project this hidden state into the 1664-dimensional ViT-G teacher space. The projected predictor intercepts the computation **before** that original output projection.

---

## 4. New 384-to-768 correction head

Discard the original

$$
384\rightarrow1664
$$

distillation head for the new forecasting objective.

Introduce one shared linear correction head

$$
\boxed{
\Pi_{\Delta}:\mathbb R^{384}\rightarrow\mathbb R^{768}.
}
$$

For each future patch,

$$
\boxed{
\Delta\hat Z_{i+1}(p)
=
\Pi_{\Delta}\left(V_i(p)\right).
}
$$

The same projection is used for future positions that were predictor context and future positions that were predictor targets.

This is important. The output has one definition everywhere:

$$
\boxed{
\Delta\hat Z(p)
=
\text{learned correction relative to the deterministic proposal at }p.
}
$$

Future context and target hidden states are scattered back into their original $24\times78$ spatial locations before forming the complete correction field.

Past/history predictor outputs are discarded.

---

## 5. Final future prediction

The predicted static future state is

$$
\boxed{
\hat Z_{i+1}
=
Z_{i+1}^{0}
+
\Delta\hat Z_{i+1}.
}
$$

Substituting the deterministic proposal gives the central model equation:

$$
\boxed{
\hat Z_{i+1}
=
S_{i+1}\odot Z_{i\rightarrow i+1}^{\mathrm{warp}}
+
(1-S_{i+1})\odot Z_i
+
\Pi_{\Delta}(V_i).
}
$$

The corresponding oracle correction is

$$
\boxed{
\Delta Z_{i+1}^{\star}
=
Z_{i+1}^{\star}-Z_{i+1}^{0}.
}
$$

The model is therefore learning to answer:

> Given the complete available video history and partial geometrically projected future, what must be changed relative to the deterministic warp-plus-copy proposal to obtain the true future static V-JEPA state?

---

## 6. Interpretation of the residual

The correction is deliberately **not** claimed to be optical flow or pure semantic motion.

Its meaning is simpler and directly measurable:

$$
\boxed{
\Delta\hat Z
=
\text{the learned intervention over the zero-learning baseline}.
}
$$

The required correction naturally differs between regions.

### 6.1 Correct static warp

If geometry is already accurate,

$$
Z_{i+1}^{0}(p)
\approx
Z_{i+1}^{\star}(p),
$$

then ideally

$$
\Delta\hat Z_{i+1}(p)
\approx0.
$$

### 6.2 Dense but incorrect dynamic warp

For a moving vehicle, geometry may provide a valid but incorrect patch.

Then

$$
S_{i+1}(p)=1
$$

but

$$
Z_{i+1}^{0}(p)
\not\approx
Z_{i+1}^{\star}(p),
$$

so the predictor must produce a non-zero correction.

### 6.3 Missing future patch

If geometry cannot provide the future patch,

$$
S_{i+1}(p)=0,
$$

and therefore

$$
Z_{i+1}^{0}(p)=Z_i(p).
$$

The correction must transform this zero-motion prior into the correct future representation.

The 768-D correction field is therefore heterogeneous in function, but this is not a conceptual problem. Its interpretation is always relative to the same deterministic baseline.

The 384-D field $V_i$ can be described more loosely as a **temporal transport/completion latent**: it is the learned spatiotemporal representation from which the correction is decoded.

---

## 7. Why no second spatial residual network is required initially

A direct $384\rightarrow768$ linear head may appear too simple because a moving object must change spatial location.

However, the spatial and temporal mixing is already performed inside the V-JEPA predictor.

Before $\Pi_\Delta$ is applied, each future hidden state $V_i(p)$ has passed through 12 layers of joint self-attention over the available context and target tokens. It can therefore depend on information from other spatial and temporal locations.

The new projection does not need to perform transport itself. Its job is only to decode the already-contextualized 384-D state into the static 768-D correction space.

This gives the simplest useful first experiment:

$$
\boxed{
\text{pretrained spatiotemporal reasoning}
+
\text{one new linear correction head}.
}
$$

If this proves insufficient, an explicit correspondence, deformable-attention or transport module can be introduced later.

---

## 8. Context and target positions

The predictor contains two kinds of future positions.

### Future context

These are future video tubelets retained because the deterministic RGB projection provided sufficient support.

They enter the video encoder as real context tokens and emerge from the predictor with contextualized 384-D hidden states.

### Future targets

These are tubelets removed before the video context transformer because their projected future RGB is insufficient.

They enter the predictor as learned 384-D mask tokens and are filled through joint self-attention.

Both subsets require correction outputs.

After predictor normalization, obtain

$$
V_{i,C}
\in
\mathbb R^{N_C^{\mathrm{future}}\times384},
$$

and

$$
V_{i,M}
\in
\mathbb R^{N_M^{\mathrm{future}}\times384}.
$$

Apply the same head:

$$
\Delta\hat Z_C
=
\Pi_\Delta(V_{i,C}),
$$

$$
\Delta\hat Z_M
=
\Pi_\Delta(V_{i,M}),
$$

and scatter both into the complete future grid.

A context patch is never frozen simply because it was geometrically valid. This is what permits correction of moving vehicles inside dense splat regions.

---

## 9. Temporal tubelet alignment

The main architectural detail that must be handled carefully is temporal alignment.

V-JEPA video tokens span two frames. A clip

$$
[I_0,I_1,I_2,I_3,\ldots]
$$

produces temporal planes corresponding to

$$
(I_0,I_1),
\quad
(I_2,I_3),
\quad\ldots
$$

Therefore, a hidden field intended to represent the transition

$$
i\rightarrow i+1
$$

must come from a tubelet aligned with

$$
(I_i,I_{i+1})
$$

or, in the predictor pathway,

$$
(I_i,\bar I_{i+1}).
$$

If consecutive transition fields are required for

$$
t\rightarrow t+1
$$

and

$$
t+1\rightarrow t+2,
$$

the standard non-overlapping tubelet tokenizer requires two temporally staggered predictor passes.

The implementation therefore assembles two eight-frame streams from the five observed frames and the four source-relative warps:

$$
\mathcal S_1
=
[I_{t-3},I_{t-2},I_{t-1},I_t,\bar I_{t+1},\bar I_{t+2},\bar I_{t+3},\bar I_{t+4}],
$$

$$
\mathcal S_2
=
[I_{t-4},I_{t-3},I_{t-2},I_{t-1},I_t,\bar I_{t+1},\bar I_{t+2},\bar I_{t+3}].
$$

Their third and fourth tubelet planes provide the four adjacent transition fields:

| Field | Stream | Tubelet |
|---|---|---|
| $V_t$ | $\mathcal S_2$, plane 2 | $(I_t,\bar I_{t+1})$ |
| $V_{t+1}$ | $\mathcal S_1$, plane 2 | $(\bar I_{t+1},\bar I_{t+2})$ |
| $V_{t+2}$ | $\mathcal S_2$, plane 3 | $(\bar I_{t+2},\bar I_{t+3})$ |
| $V_{t+3}$ | $\mathcal S_1$, plane 3 | $(\bar I_{t+3},\bar I_{t+4})$ |

Only the first transition can pair a real frame with a warp; later transitions pair two warps because their earlier frame is itself unobserved.

Both frames of a future tubelet share the validity mask of the pair's second frame, i.e. $S_{t+k}$ for the plane supplying $V_{t+k-1}$. The context/target token partition of that plane therefore coincides with the sets $C_{t+k}$ and $M_{t+k}$ used by the deterministic proposal and the loss. History planes are fully observed context.

---

## 10. Four-step daisy-chained rollout

The first model predicts four future static states, $\hat Z_{t+1},\ldots,\hat Z_{t+4}$.

Every deterministic warp is generated directly from the latest observed RGB-D frame, so each proposal blends the horizon-$k$ warp with the most recent state. With

$$
\hat Z_t:=Z_t^\star,
$$

each step $k=1,\ldots,4$ constructs

$$
\boxed{
Z_{t+k}^{0}
=
S_{t+k}\odot Z_{t\rightarrow t+k}^{\mathrm{warp}}
+
(1-S_{t+k})\odot\hat Z_{t+k-1},
}
$$

takes the transition hidden field

$$
V_{t+k-1}
\in
\mathbb R^{24\times78\times384}
$$

from the staggered streams of Section 9, and predicts

$$
\boxed{
\hat Z_{t+k}
=
Z_{t+k}^{0}
+
\Pi_\Delta\left(V_{t+k-1}\right).
}
$$

Later predictions therefore depend autoregressively on earlier ones only where deterministic geometry cannot provide a usable future patch.

The hidden fields themselves are computed once, from RGB history and warp evidence alone; the predictor never consumes its own latent predictions. Gradients nevertheless reach earlier corrections through the copy branch of the proposal chain.

---

## 11. Reference path

The original 1664-dimensional ViT-G distillation target is not required by this new objective.

The reference future latents are simply

$$
Z_{t+k}^{\star}
=
E_{B,\mathrm{frozen}}^{\mathrm{img}}(I_{t+k}),
\qquad
k=1,\ldots,4.
$$

The same frozen image encoder also produces the warped static latents used in $Z^0$.

Ground-truth future video encodings are therefore unnecessary for the core projected-predictor loss.

---

## 12. Loss

The primary loss is patchwise $L_1$ distance in the frozen static ViT-B latent space, with the geometrically valid and missing regions normalised separately so the typically larger valid area cannot drown out the completion objective.

With the channel-mean patch error

$$
\ell(k,p)
=
\frac{1}{768}
\left\|
\hat Z_{t+k}(p)-Z_{t+k}^{\star}(p)
\right\|_1,
$$

the training objective pools all four horizons and averages each region independently:

$$
\boxed{
\mathcal L
=
\operatorname*{mean}_{\{(k,p)\,:\,p\in C_{t+k}\}}
\ell(k,p)
+
\operatorname*{mean}_{\{(k,p)\,:\,p\in M_{t+k}\}}
\ell(k,p).
}
$$

No distance weighting is applied within the valid region, and there are no per-horizon weights; pooling patches across horizons implicitly weights each horizon by its region size (longer horizons contribute more missing patches to the second term).

Because

$$
\hat Z=Z^0+\Delta\hat Z,
$$

this directly trains the model toward the oracle correction

$$
\Delta Z^\star=Z^\star-Z^0.
$$

### Diagnostics

The context and masked terms are logged separately, together with the unpartitioned per-horizon means

$$
\mathcal L_k
=
\frac{1}{N}
\sum_{p=1}^{N}
\ell(k,p),
\qquad
k=1,\ldots,4,
$$

which are health metrics only, not optimisation terms. A validation pass before the first optimizer step records the exact B1 warp-plus-copy baseline under the identical pipeline, since the zero-initialised head gives $\hat Z=Z^0$.

---

## 13. Initialization and fine-tuning

The forecasting model is initialized from the released V-JEPA 2.1 ViT-B checkpoint.

Pretrained components retained are:

- video tokenizer;
- ViT-B video encoder;
- predictor $768\rightarrow384$ input projection;
- learned mask tokens;
- 12 predictor transformer blocks;
- predictor LayerNorm.

The only new module is

$$
\Pi_\Delta:384\rightarrow768.
$$

Initialize its weights and bias to zero:

$$
W_\Delta=0,
\qquad
b_\Delta=0.
$$

At optimization step zero,

$$
\Delta\hat Z=0,
$$

so

$$
\boxed{
\hat Z=Z^0.
}
$$

Training therefore begins exactly from the deterministic warp-plus-copy baseline.

The encoder remains entirely frozen, in image mode and in video mode: the same ViT-B weights define the regression-target space consumed by the frozen dense decoders, so fine-tuning the video pathway would move the target space itself. Only the predictor body is post-trained, with a lower learning rate than the newly initialized correction head. The predictor's mask tokens are the exception: the released checkpoint's eight mask tokens are degenerate/interchangeable, so they are effectively fresh parameters and share the correction head's learning rate.

RoPE position handling follows the released configuration: the encoder rescales patch positions onto the fixed pretraining span (`interpolate_rope=True`), while the predictor uses raw integer positions (`interpolate_rope=False`), preserving its trained local position metric on the non-square KITTI grid at the cost of extrapolated far-horizontal offsets, which fine-tuning adapts.

---

## 14. Interpretability

The simple architecture retains a useful form of interpretability because the zero-learning counterfactual is explicit.

### 14.1 Where did learning intervene?

Visualize

$$
r(p)
=
\left\|\Delta\hat Z(p)\right\|_2.
$$

This gives a patchwise heatmap of where the learned predictor chose to depart from deterministic geometry and copy-forward.

### 14.2 Where was correction actually required?

Compute

$$
r^\star(p)
=
\left\|Z^\star(p)-Z^0(p)\right\|_2.
$$

Comparing $r$ and $r^\star$ shows whether the learned model intervened in the regions where the deterministic proposal was actually wrong.

### 14.3 Was the correction direction sensible?

Define

$$
\Delta Z^\star(p)
=
Z^\star(p)-Z^0(p).
$$

Then compare

$$
\operatorname{cos}
\left(
\Delta\hat Z(p),
\Delta Z^\star(p)
\right).
$$

### 14.4 Dense-decoder effect

For the frozen depth or semantic decoder $D$, compare

$$
D(Z^0),
\qquad
D(\hat Z),
\qquad
D(Z^\star).
$$

This directly shows what the learned component changed relative to the deterministic baseline.

The first architecture does not expose explicit patch-to-patch correspondence. If later experiments require stronger transport interpretability, a correspondence or deformable-attention bottleneck can be introduced after establishing this simpler baseline.

---

## 15. Baselines and ablations

The system naturally provides a clear experimental ladder.

### B0: copy last

$$
\hat Z_{i+1}=Z_i.
$$

### B1: deterministic warp plus copy-forward

$$
\boxed{
\hat Z_{i+1}=Z_{i+1}^{0}.
}
$$

### B2: projected predictor

$$
\boxed{
\hat Z_{i+1}
=
Z_{i+1}^{0}
+
\Pi_\Delta(V_i).
}
$$


---

## 16. Final formulation

The projected predictor is intentionally minimal:

$$
\boxed{
\hat Z_{i+1}
=
\underbrace{
S_{i+1}\odot Z_{i\rightarrow i+1}^{\mathrm{warp}}
+
(1-S_{i+1})\odot Z_i
}_{\text{deterministic warp + copy-forward proposal}}
+
\underbrace{
\Pi_\Delta(V_i)
}_{\text{learned V-JEPA correction}}.
}
$$

The pretrained V-JEPA video encoder and predictor see the complete available spatiotemporal context and are fine-tuned to produce a useful 384-D temporal representation $V_i$.

A single new shared linear projection converts that representation into a 768-D correction

This gives the first model three desirable properties:

- **simplicity**: only one new $384\rightarrow768$ head is required;
- **use of prior knowledge**: the expensive spatiotemporal reasoning begins from pretrained V-JEPA weights;
- **interpretability relative to a counterfactual**: setting the learned correction to zero exactly recovers the deterministic baseline.

