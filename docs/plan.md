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
3. Can predicted representations be converted into useful future BEV maps for collision-risk and trajectory ranking?

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

Use one canonical final-layer V-JEPA 2.1 feature tensor as the system state. The world model and both probes must consume exactly the same representation. 

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
