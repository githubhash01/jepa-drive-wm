# MSc World Model Implementation Plan

## Baseline system

Build a deterministic latent world model mostly inspired by **Back to the Features**

- Encode four historical frames with a frozen visual encoder.
- Use the latent patch tokens as historical memory.
- Create learned future patch queries.
- Update the future queries through cross-attention to the historical memory.
- Encode the candidate ego motion with a simple MLP.
- Inject the action condition with a simple residual conditioning block.
- Project the future queries into the predicted next latent frame.
- Train with direct latent regression

## Initial evaluation

Test whether the model:

- outperforms a copy-last baseline;
- uses the supplied ego motion meaningfully;
- generalizes to held-out KITTI sequences;
- supports short autoregressive rollouts;
- remains computationally efficient; 
- remains decodable with frozen probes 

## Deferred extensions

Only after the baseline works, consider:

- AdaLN action conditioning;
- self-attention among future queries;
- Fourier action encoding;
- historical ego-motion inputs;
- rollout losses or consistency objectives.

The priority is one clear, working baseline before adding architectural complexity.