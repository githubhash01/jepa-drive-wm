"""
Shared training helpers, so the depth / semantic / world-model loops agree on:

- an iteration budget instead of a fixed epoch count (the proven recipe: the old
  DPT probes trained for a fixed ~50-70k forward/backward passes with a cosine
  schedule, not N full passes over the data);
- bf16 autocast (the old pipeline used AMP + SDPA; it is ~1.5-2x faster than fp32
  and, unlike fp16, bf16 needs no GradScaler);
- a linear warmup into cosine decay, stepped once per *optimizer* step;
- one wandb naming convention (`init_run`), so three runs are told apart from
  the runs table without opening them.

`MAX_ITERS` everywhere counts forward/backward passes (micro-steps), matching the
`iter` counter in the old train logs. With GRAD_ACCUM micro-steps per optimizer
step, there are MAX_ITERS // GRAD_ACCUM optimizer steps, which is what the
schedule is sized against.
"""
from typing import Sequence

import torch
import wandb

from jepa_drive_wm.data.splits import KITTISplit

# One project for the whole thesis; `job_type` separates the models within it.
WANDB_PROJECT = "jepa-drive-wm"

# bf16 autocast: Ampere+ (RTX 30/40) runs it fast and it has fp32's exponent
# range, so no loss scaling is needed. Ada (4070 Ti SUPER) is happy here.
AMP_DTYPE = torch.bfloat16


def infinite(loader):
    """Yield batches forever, restarting the loader (and reshuffling) each pass."""
    while True:
        yield from loader


def build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_opt_steps: int,
    warmup_opt_steps: int,
    warmup_start_factor: float = 1e-2,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup from warmup_start_factor*lr up to lr, then cosine to ~0.

    Stepped once per optimizer step (i.e. every GRAD_ACCUM micro-steps), so
    `total_opt_steps` and `warmup_opt_steps` are in optimizer-step units.
    """
    warmup_opt_steps = max(1, warmup_opt_steps)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_opt_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_opt_steps - warmup_opt_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_opt_steps]
    )


def autocast(device: torch.device):
    """bf16 autocast on cuda; a no-op elsewhere (cpu autocast would just slow us)."""
    return torch.autocast(
        device_type=device.type, dtype=AMP_DTYPE, enabled=device.type == "cuda"
    )


def init_run(
    *,
    job_type: str,
    name: str,
    tags: Sequence[str],
    config: dict,
    split: KITTISplit,
    mode: str = "online",
):
    """Start a wandb run under the shared naming convention.

        project   jepa-drive-wm
        group     split-<name>    -- runs grouped by the split they are comparable on
        job_type  depth_decoder | semantic_decoder | ac_predictor
        name      <caller's descriptive name>-<4 chars of the run id>
        tags      caller's tags, plus the job type and the split

    wandb's auto-generated names ("dandy-sunset-42") make three runs of three
    different models indistinguishable in the runs table, so `name` should encode
    whatever actually varies between runs of *this* model. The run-id suffix keeps
    repeat launches of one config visually distinct.

    The split is folded into the config here rather than by each caller, so every
    run records which sequences it trained and validated on. The `define_metric`
    calls pin `train/*` and `val/*` to the `iter` counter (micro-steps), so
    validation points land on the same x-axis as training despite being logged
    far less often.
    """
    run = wandb.init(
        project=WANDB_PROJECT,
        job_type=job_type,
        group=f"split-{split.name}",
        tags=[*tags, job_type, f"split-{split.name}"],
        config={
            **config,
            "amp_dtype": str(AMP_DTYPE).removeprefix("torch."),
            "split": split.name,
            "train_sequences": list(split.train_sequences),
            "validation_sequences": list(split.validation_sequences),
            "test_sequences": list(split.test_sequences),
        },
        mode=mode,
    )
    run.name = f"{name}-{run.id[:4]}"

    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")
    wandb.define_metric("val/*", step_metric="iter")
    return run
