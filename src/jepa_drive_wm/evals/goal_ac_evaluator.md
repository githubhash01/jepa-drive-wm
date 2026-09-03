I agree. This is a much better fit for the scope of the model and the time available. It gives you **three distinct answers**:

1. How accurately does the model predict future V-JEPA latents?
2. Do those predicted latents preserve useful depth and semantic information?
3. Does changing the action input change—and ideally improve—the prediction?

That is enough for a focused evaluation. The evaluator should now be shaped around the write-up, rather than producing every metric it is capable of calculating.

The four-step protocol is already consistent with the evaluator’s defaults: four context frames and four autoregressive predictions, corresponding to `0.8` s for the `0.2` s-step model and `2.0` s for the `0.5` s-step model.

One important presentation rule is that the fourth prediction of the two models should **not** be directly compared as though they represent the same forecasting problem:

- `0.2` s model, step 4: `t+0.8` s
- `0.5` s model, step 4: `t+2.0` s

The models should primarily be discussed independently: how each degrades across its own four rollout horizons.

---

# Recommended final structure

## `\subsubsection*{Latent Prediction Error}`

For each model:

- provide four cached V-JEPA context latents;
- predict four future latents autoregressively;
- calculate the mean absolute difference between each predicted latent and its cached target latent;
- average over the latent spatial dimensions, channels, and test windows.

That is already how the evaluator calculates latent L1: it takes the absolute predicted–target difference and averages over `H`, `W`, and `C`, retaining one score per window and rollout step before pooling across the dataset.

### `0.2` s-step model

| Prediction horizonAutoregressive latent L1 ↓Copy-last latent L1 ↓ |   |   |
| ----------------------------------------------------------------- | - | - |
| `t+0.2` s                                                         | … | … |
| `t+0.4` s                                                         | … | … |
| `t+0.6` s                                                         | … | … |
| `t+0.8` s                                                         | … | … |

### `0.5` s-step model

| Prediction horizonAutoregressive latent L1 ↓Copy-last latent L1 ↓ |   |   |
| ----------------------------------------------------------------- | - | - |
| `t+0.5` s                                                         | … | … |
| `t+1.0` s                                                         | … | … |
| `t+1.5` s                                                         | … | … |
| `t+2.0` s                                                         | … | … |

## What to do with persistence

I recommend retaining **only the copy-last baseline**, as one additional column in these tables.

It is not another large evaluation branch. It answers one basic question that raw L1 cannot answer:

> Did the learned predictor forecast the latent more accurately than simply assuming that the final context latent remained unchanged?

V-JEPA latents of adjacent driving frames may naturally be similar. Consequently, a numerically small prediction error does not necessarily mean that the model learned useful dynamics. If copying `z_t` produces the same or lower error, then the predictor has not improved over temporal persistence.

You do **not** need:

- teacher-forced results;
- copy-previous results;
- rollout/copy ratios;
- a separate persistence subsection;
- persistence decoding for depth and semantics.

The current evaluator calculates all of those latent variants. For your final scope, I would retain only:

```text
rollout
copy_last
rollout_zero_action
```

and add action sensitivity.

If you are absolutely determined not to show persistence in the main table, it should at least remain in `metrics.json` and be summarised in one sentence such as:

> The learned rollout remained below the copy-last baseline at all four prediction horizons.

But one extra table column is the clearer option.

---

# `\subsubsection*{Decoded Performance}`

This section should answer whether task-relevant information remains recoverable from the predicted latent. It does not need to become a second full evaluation of the readout decoders.

The current evaluator decodes three latent sources—target, copy-last and predicted—and calculates a broad collection of metrics. You can reduce that to:

- decode the **autoregressively predicted latent**;
- compare it with the pseudolabel belonging to that future frame;
- report one selected depth metric and one selected semantic metric.

Assuming the depth and semantic decoder performance on true cached latents has already been established in the preceding readout evaluation, simply cross-reference that section. You do not need to reproduce the entire target-latent decoder baseline here.

A suitable methods sentence would be:

> At each rollout horizon, the predicted latent was passed through the frozen depth and semantic readouts. The resulting predictions were evaluated against the corresponding FoundationStereo depth and OneFormer semantic pseudolabels.

## Depth

Use only **non-sky AbsRel**.

### `0.2` s-step model

| Prediction horizonPredicted-latent non-sky AbsRel ↓ |   |
| --------------------------------------------------- | - |
| `t+0.2` s                                           | … |
| `t+0.4` s                                           | … |
| `t+0.6` s                                           | … |
| `t+0.8` s                                           | … |

### `0.5` s-step model

| Prediction horizonPredicted-latent non-sky AbsRel ↓ |   |
| --------------------------------------------------- | - |
| `t+0.5` s                                           | … |
| `t+1.0` s                                           | … |
| `t+1.5` s                                           | … |
| `t+2.0` s                                           | … |

The accompanying discussion only needs to answer:

- Does AbsRel increase with rollout horizon?
- Is the increase gradual or sharp?
- Does the decoded output remain broadly plausible at the final horizon?
- Are the errors concentrated around object boundaries, distant geometry, or moving vehicles?

### Qualitative depth figure

Make it exactly as you described:

| Horizon 1Horizon 2Horizon 3Horizon 4 |       |       |       |       |
| ------------------------------------ | ----- | ----- | ----- | ----- |
| FoundationStereo pseudolabel         | image | image | image | image |
| Depth decoded from predicted latent  | image | image | image | image |

Use the same depth range and colour scale for every cell. Otherwise, independently normalised images can make poor predictions look deceptively similar.

The current qualitative figure contains camera images and decoder outputs from target, copied, and predicted latents. Those extra rows can be removed.

I would select one deterministic, representative test window rather than manually choosing the most visually appealing output. The evaluator already selects qualitative windows deterministically from a fixed anchor grid, so that behaviour can be retained.

## Semantic segmentation

Use only **planning-group mIoU**.

### `0.2` s-step model

| Prediction horizonPredicted-latent planning-group mIoU ↑ |   |
| -------------------------------------------------------- | - |
| `t+0.2` s                                                | … |
| `t+0.4` s                                                | … |
| `t+0.6` s                                                | … |
| `t+0.8` s                                                | … |

### `0.5` s-step model

| Prediction horizonPredicted-latent planning-group mIoU ↑ |   |
| -------------------------------------------------------- | - |
| `t+0.5` s                                                | … |
| `t+1.0` s                                                | … |
| `t+1.5` s                                                | … |
| `t+2.0` s                                                | … |

The discussion can remain similarly direct:

- Does planning-group mIoU decline over the rollout?
- Which planning-relevant regions visually deteriorate first?
- Are road boundaries and traffic participants retained?
- Does the model preserve coarse scene organisation while losing fine boundaries?

### Qualitative semantic figure

| Horizon 1Horizon 2Horizon 3Horizon 4    |       |       |       |       |
| --------------------------------------- | ----- | ----- | ----- | ----- |
| OneFormer pseudolabel                   | image | image | image | image |
| Semantics decoded from predicted latent | image | image | image | image |

Use the identical class-colour mapping for both rows.

You could create one depth figure containing a panel for each timestep model and one equivalent semantic figure. That gives two main qualitative figures rather than four separate figures.

---

# `\subsubsection*{Action Sensitivity}`

This section should distinguish **whether actions help** from **whether actions affect the output at all**.

For each test window, run the same autoregressive rollout twice:

1. with the recorded ego-motion actions;
2. with every future action replaced by zero.

The current evaluator already calculates the first two quantities: real-action rollout L1 and zero-action rollout L1.

Add the direct action-sensitivity measurement:

```math
S_k = \frac{1}{HWC} \left\| \hat{z}_{t+k}^{\,\mathrm{real}} - \hat{z}_{t+k}^{\,\mathrm{zero}} \right\|_1.
```

After averaging across test windows, `S_k` measures how much the prediction changes when the action input is removed.

Also calculate the error difference:

```math
\Delta L_k = L^{\mathrm{zero}}_k-L^{\mathrm{real}}_k.
```

Interpretation:

- `\Delta L_k>0`: supplying the real action improves target prediction.
- `\Delta L_k\approx0`: no measurable performance benefit from the real action.
- `\Delta L_k<0`: the zero-action rollout performs better.
- `S_k\approx0`: the model barely changes its prediction when actions are removed.
- `S_k>0`, but `\Delta L_k\approx0`: the model responds to actions, but that response does not improve accuracy.
- `S_k>0` and `\Delta L_k>0`: the model responds to the action and the response is useful.

### Suggested action table

#### `0.2` s-step model

| HorizonReal-action L1 ↓Zero-action L1 ↓Zero − real L1 ↑Action sensitivity |   |   |   |   |
| ------------------------------------------------------------------------- | - | - | - | - |
| `t+0.2` s                                                                 | … | … | … | … |
| `t+0.4` s                                                                 | … | … | … | … |
| `t+0.6` s                                                                 | … | … | … | … |
| `t+0.8` s                                                                 | … | … | … | … |

Then repeat for the `0.5` s-step model.

There is some repetition of the real-action L1 from the latent-prediction table, but that is acceptable: it makes the action table self-contained and there are only four rows.

I would call this section **Action Sensitivity**, not “Action-Conditioning Effectiveness.” It is a test-time ablation and supports careful statements about whether the predictions changed or improved. It does not, by itself, prove a causal understanding of vehicle control.

---

# Resulting evaluator specification

The streamlined evaluator should calculate:

## Latent quantities

```text
autoregressive rollout L1
copy-last L1
zero-action rollout L1
real-versus-zero prediction sensitivity
```

## Decoded quantities

```text
predicted-latent non-sky depth AbsRel
predicted-latent planning-group semantic mIoU
```

## Qualitative outputs

```text
depth:      pseudolabel vs predicted-latent decode over four horizons
semantics:  pseudolabel vs predicted-latent decode over four horizons
```

## Outputs

```text
metrics.json
summary.md containing the exact report tables
depth qualitative figure
semantic qualitative figure
```

The shared evaluation code already provides checkpoint loading and machine-readable JSON output, so that infrastructure should remain unchanged.

---

# What can be removed from the current evaluator

For this final evaluation, I would remove or stop reporting:

- teacher-forced prediction;
- copy-previous baseline;
- per-sequence breakdowns;
- RMSE;
- depth `\delta_1`;
- vehicle-region AbsRel;
- 19-class semantic mIoU;
- drivable IoU;
- boundary IoU;
- traffic-participant IoU;
- car IoU;
- target-latent and copy-latent decoding;
- large multi-metric comparison plots;
- camera and extra decoder rows in the qualitative figures.

There is no need to alter the depth and semantic accumulator implementations merely because they internally compute more metrics. The quickest and safest approach is to let them continue producing their existing summaries, but extract only:

```python
depth_summary["non-sky"]["absrel"]
semantic_summary["planning_group_miou"]
```

for the report output.

## Final judgement on copy-last

Keep it, but keep it small.

It should be **one column in the latent tables**, not a subsection, not another decoded experiment, and not a large figure. It provides the minimum evidence needed to say that the learned dynamics outperform doing nothing. Everything else in your proposed structure is already sufficiently focused.