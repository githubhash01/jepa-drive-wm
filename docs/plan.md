# Current Project Plan & Timeline

## Working Title

**Action-Conditioned Future V-JEPA 2.1 Representation Prediction for Self-Supervised BEV Scene Forecasting**

## Project Goal

Build a driving world model that predicts future dense, front-view V-JEPA 2.1 representations under proposed ego-motion trajectories. The world model is trained only from video, ego-motion and future V-JEPA representations. It does not use depth, semantic, occupancy or BEV labels.

At inference, frozen depth and semantic probes decode the predicted future representations. A fixed Lift-Splat-Shoot (LSS) module then converts those predictions into future semantic or occupancy BEV maps for trajectory evaluation and planning.

The central systems claim is that a scalable, self-supervised latent dynamics model can support planning-oriented future BEV prediction without training the world model with explicit BEV supervision.

## Core Research Questions

1. Can future native V-JEPA 2.1 representations be predicted accurately enough that frozen depth and semantic probes remain useful?
2. Does spatially dependent ego-motion conditioning improve counterfactual future prediction over a global action embedding?
3. Does clean-latent diffusion with DDIM work better than deterministic regression for high-dimensional V-JEPA representations?
4. Can predicted representations be converted into useful future BEV maps for collision-risk and trajectory ranking?

## Main Contribution

The project is inspired by DINO-Foresight, but differs in four important ways:

- It predicts the native final V-JEPA 2.1 representation rather than a PCA-compressed concatenation of intermediate DINO features.
- It conditions predictions on hypothetical future ego motion.
- It uses spatially dependent camera-motion conditioning rather than only a global action vector.
- It converts predicted front-view representations into future BEV maps through frozen probes and known geometry.

The world model remains task-label-free. Depth and semantic supervision are confined to separately trained probes.

## System Overview

### World-Model Training

```text
past RGB frames
    -> frozen V-JEPA 2.1 encoder
    -> clean context representations

future RGB frames
    -> frozen V-JEPA 2.1 encoder
    -> clean target representations

context representations + candidate ego trajectory + noised future representations
    -> action-conditioned diffusion transformer
    -> predicted clean future V-JEPA representations
```

Training supervision is only the real future V-JEPA representation.

### Inference and Planning

```text
past RGB frames + candidate ego trajectory
    -> future V-JEPA representation prediction
    -> frozen depth probe
    -> frozen semantic probe
    -> LSS projection
    -> future semantic/occupancy BEV sequence
    -> trajectory cost and ranking
```

The planner does not require a future goal image. Route progress can come from a waypoint, route centreline or map command, while the predicted BEV supplies collision, occupancy and scene-risk costs.

## Core Components

### 1. Native V-JEPA 2.1 Representation

Use one canonical final-layer V-JEPA 2.1 feature tensor as the system state. The world model and both probes must consume exactly the same representation, including:

- encoder checkpoint;
- input resolution and preprocessing;
- final LayerNorm convention;
- patch-token selection and ordering;
- temporal/frame extraction procedure;
- tensor normalisation.

The encoder remains frozen. Context and target features should be cached where practical to reduce training cost.

The initial implementation should encode frames independently or with a strictly past-only window to avoid future information leaking through a bidirectional video encoder.

### 2. Offline Depth Probe

Train a frozen dense decoder from the native V-JEPA representation to metric depth or a depth-bin distribution.

Requirements:

- good performance on real V-JEPA features;
- calibrated depth outputs suitable for geometric lifting;
- a fixed interface that can also consume predicted features;
- evaluation on both real and world-model-predicted representations.

The probe is trained separately and is not used to supervise the primary world model.

### 3. Offline Semantic Probe

Train a frozen dense decoder from the same native V-JEPA representation to semantic class probabilities.

Requirements:

- preserve spatial resolution at the V-JEPA patch grid or upsample consistently;
- expose class probabilities rather than only hard labels;
- evaluate degradation when applied to predicted future representations;
- use the same fixed probe for oracle and predicted-feature experiments.

The probe is also excluded from the primary world-model loss.

### 4. Lift-Splat-Shoot BEV Module

Use predicted depth distributions and semantic probabilities to lift front-view features into 3D and splat them into an ego-centric BEV grid.


### 5. Action-Conditioned V-JEPA World Model

The model learns

\[
p_\theta\left(Z_{t+1:t+H}\mid Z_{t-K+1:t}, A_{t:t+H-1}\right),
\]

where \(Z\) is the native front-view V-JEPA representation and \(A\) is a proposed ego-motion or control sequence.

The first model should predict one fixed future horizon. Once stable, extend it to several sparse horizons or a short future chunk.

## Ego-Motion Representation

Use relative ego poses or integrated vehicle controls expressed in a consistent local frame. A per-step trajectory can contain:

\[
a_i = (\Delta x_i, \Delta y_i, \Delta \psi_i, \Delta t_i).
\]

The complete sequence should be retained rather than replaced only by its final displacement, because different paths can share an endpoint while producing different observations.

Distinguish clearly between:

- commanded controls, which represent intended action;
- measured odometry, which represents realised motion.

The first implementation can use measured relative poses if command data are unavailable, but the paper must describe the model as ego-trajectory-conditioned rather than implying access to low-level controls.

## Diffusion Formulation

### Clean-Latent Prediction

Only future target representations are corrupted:

\[
Z_\tau = \sqrt{\bar\alpha_\tau}Z_0 + \sqrt{1-\bar\alpha_\tau}\epsilon,
\qquad \epsilon \sim \mathcal N(0,I).
\]

The model predicts the clean representation directly:

\[
\hat Z_0 = f_\theta(Z_\tau, Z_{\mathrm{context}}, A, R, \tau).
\]

The primary objective is a masked, normalised clean-target loss:

\[
\mathcal L_{\mathrm{latent}} = \|\hat Z_0 - Z_0\|_2^2,
\]

computed over valid spatial and temporal tokens.

Use fixed training-set statistics to normalise V-JEPA features. Do not clip predicted features to \([-1,1]\).

### DDIM Sampling

Use deterministic DDIM with \(\eta=0\) for the initial system. Begin with approximately 8-16 denoising steps and compare more steps only after the model is stable.

Different initial noise samples can later be used to generate multiple plausible futures for risk-sensitive planning.

## Diffusion Architecture

Use the high-dimensional dense-token architecture from RAE-NWM as the main implementation base rather than designing a DiT from scratch.

### Reused Components

From RAE-NWM:

- target-token self-attention;
- cross-attention from noisy future tokens to clean context tokens;
- action and horizon embeddings;
- CDiT backbone;
- shallow, wide DDT prediction head;
- rollout and RECON-style data infrastructure where useful.

From NWM or another tested Gaussian-diffusion implementation:

- forward noising process;
- clean-\(x_0\) prediction mode;
- cosine or equivalent noise schedule;
- timestep respacing;
- DDIM sampling.

### Initial Model Configuration

A practical starting point is:

- 2-4 past context frames;
- one future frame or fixed horizon;
- native V-JEPA patch grid;
- 8-12 CDiT blocks;
- backbone width near 768;
- target self-attention plus context cross-attention;
- two-block DDT head;
- DDT width approximately 1,536-2,048;
- global trajectory AdaLN conditioning;
- dense ray-motion embeddings added to target tokens and the DDT condition.

The wide head is important because a narrow projection can discard information needed to reconstruct high-dimensional V-JEPA tokens under Gaussian corruption.

## Training Strategy

### Stage 1: Representation and Data Validation

- Freeze and cache canonical V-JEPA features.
- Confirm that cached features reproduce probe outputs.
- Verify timestamps, camera calibration and ego-motion alignment.
- Measure feature mean, variance, norm distribution and effective dimensionality.
- Confirm that no future frames leak into context representations.

### Stage 2: Deterministic Baseline

Train an action-conditioned deterministic predictor using SmoothL1 or MSE. This is the direct analogue of an action-conditioned DINO-Foresight baseline.

It must beat:

- copying the latest feature;
- constant-velocity or simple warp baselines where applicable;
- the same model with shuffled actions.

### Stage 3: Global Action-Conditioned Diffusion

Implement clean-latent prediction and DDIM using only the global trajectory embedding. Validate:

- overfitting on a tiny subset;
- reconstruction across noise levels;
- stable DDIM trajectories;
- preserved feature norms;
- probe compatibility.

### Stage 4: Spatial Conditioning

Add per-token ray conditioning. Test fixed-context, fixed-noise action sweeps for:

- left versus right steering;
- different yaw magnitudes;
- braking versus acceleration;
- zero-motion behaviour;
- shuffled or incorrect trajectories.

### Stage 5: Multi-Horizon Prediction and Rollout

Extend to sparse future horizons or a short jointly predicted chunk. If autoregressive rollout is used, train with at least one model-generated context step to reduce exposure bias.

Avoid predicting all horizons as residuals from one unchanged current-frame anchor. Each future horizon should have its own pose condition, and autoregressive predictions should be re-anchored on the latest predicted state.

### Stage 6: Inference-Time BEV and Planning

Run frozen depth and semantic probes on predicted representations, then LSS to BEV. Use the resulting future BEV sequence to score candidate trajectories.

A trajectory cost can combine:

- collision or occupied-area overlap with the ego footprint;
- unsafe semantic classes;
- route or waypoint progress;
- uncertainty or unknown-space penalties;
- acceleration, curvature and jerk penalties.

Without a route-progress term, the safest policy may be to stop, so BEV risk is only one component of the planner objective.

## Evaluation

### Representation Metrics

- latent MSE;
- token cosine similarity;
- error versus prediction horizon;
- spatial-neighbourhood or token-affinity preservation;
- token norm and channel-statistics drift.

### Probe Metrics

Apply the same frozen probes to real and predicted future features.

- semantic mIoU and per-class accuracy;
- metric depth error;
- probe confidence and entropy;
- performance degradation relative to real future V-JEPA features.

### BEV Metrics

- semantic BEV mIoU;
- occupancy precision, recall and IoU;
- visibility-aware evaluation;
- error versus future horizon;
- comparison with the oracle probe-derived future BEV.

### Planning and Controllability Metrics

- collision-risk ranking accuracy;
- dangerous-trajectory recall;
- trajectory ranking correlation;
- planning success or collision rate in simulation, if available;
- sensitivity to counterfactual ego trajectories;
- degradation when actions are shuffled;
- consistency of predicted motion direction and magnitude.

### Uncertainty Evaluation

Compare deterministic regression with multiple diffusion samples under the same candidate trajectory. Test whether stochastic prediction improves high-risk event recall or risk-sensitive trajectory ranking.

## Required Baselines

1. Copy the latest V-JEPA representation.
2. Deterministic action-conditioned native V-JEPA predictor.
3. Clean-latent DDIM with global action conditioning only.
4. Clean-latent DDIM with global plus spatial conditioning.
5. Standard output head versus shallow-wide DDT head.
6. Real future V-JEPA features through the frozen probes and LSS as an oracle ceiling.
7. Optional supervised BEV forecasting baseline, used only for comparison and not as part of the proposed training method.

## Main Ablations

- no action versus global action versus spatial action conditioning;
- deterministic regression versus clean-latent diffusion;
- final native V-JEPA state versus any compressed alternative;
- ordinary linear prediction head versus wide DDT head;
- one future horizon versus multi-horizon prediction;
- different DDIM step counts;
- latent-only loss versus optional frozen-probe consistency loss;
- amount of labelled data used to train the offline probes.

The primary model should remain latent-only. Probe-consistency supervision is a fallback or ablation because it weakens the claim that the world model is task-label-independent.
