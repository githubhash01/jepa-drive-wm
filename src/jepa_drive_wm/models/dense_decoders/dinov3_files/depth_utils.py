"""
Entirely Copied from DINOV3 
"""
import torch


def create_chmv2_mixlog_bins(min_depth, max_depth, n_bins, device):
    """
    Creates mixed log bins for the CHMv2 model.
    Bins are interpolated between linear and log distributions.

    Note: max_depth is divided by 8.0 because the CHMv2 model was trained
    with internally scaled depth values. The scaling is reversed in
    `create_outputs_with_chmv2_mixlog_norm` by multiplying by 8.0.
    """
    scaled_max_depth = max_depth / 8.0
    linear = torch.linspace(min_depth, scaled_max_depth, n_bins, device=device)
    log = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(min_depth)),
            torch.log(torch.tensor(scaled_max_depth)),
            n_bins,
            device=device,
        )
    )
    t = torch.linspace(1.0, 0.0, n_bins, device=device)
    bins = t * log + (1.0 - t) * linear
    return bins


def create_outputs_with_chmv2_mixlog_norm(
    input: torch.Tensor,
    bins: torch.Tensor,
    max_clamp_value: float = 1e-4,
    eps_shift: float = 1e-8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Converts depth bin logits to depth values using mixlog normalization, specifically
    for the CHMv2 model.
    This function implements a "soft-argmax" style depth prediction, where the output
    is a weighted sum of depth bins, with weights derived from the input logits.
    The CHMv2 model outputs values that are 8x smaller than actual depth in meters,
    so we multiply by 8.0 at the end.

    Args:
        input: Raw logits from the decoder head.
        bins: Depth bin centers created by `create_chmv2_mixlog_bins`.
        max_clamp_value: Maximum value for the positive shift
        eps_shift: Epsilon added to shift to prevent division by zero
        eps: Epsilon for numerical stability in division and final clamping

    Returns:
        Depth map in meters (after x8.0 to the outputs)
    """
    y = torch.relu(input)

    # Ensure strictly positive values by adding a small shift
    m = y.amin(dim=1, keepdim=True)
    shift = (-m).clamp_min(0.0).clamp_max(max_clamp_value) + eps_shift
    y_pos = y + shift

    # Normalize to get weights that sum to 1 (linear normalization)
    # Similar to softmax but without the exponential to preserve relative magnitudes
    denom = y_pos.sum(dim=1, keepdim=True)
    denom = torch.nan_to_num(denom, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
    weights = y_pos / denom  # (N, n_bins, H, W), sums to 1 along dim=1

    # Compute weighted sum of depth bins (soft-argmax style depth prediction)
    bins_broadcast = bins.view(1, -1, 1, 1).clamp_min(eps)  # (1, n_bins, 1, 1) to match the weights shape
    output = (weights * bins_broadcast).sum(dim=1, keepdim=True).clamp_min(eps)  # (N, 1, H, W)

    # Scale back to meters (reverse the /8.0 applied in `create_chmv2_mixlog_bins``)
    output = output * 8.0

    return output

class FeaturesToDepth(torch.nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=80,
        bins_strategy="linear",
        norm_strategy="linear",
    ):
        """
        Module which converts a feature maps into a depth map

        Args:
        min_depth (float): minimum depth, used to calibrate the depth range
        max_depth (float): maximum depth, used to calibrate the depth range
        bins_strategy (str): Choices are 'linear', 'log', or 'chmv2_mixlog'. The first two are for Uniform or Scale Invariant distributions
                             for depth bins. See AdaBins [1] for more details.
                             "chmv2_mixlog" uses customized mixed log bins for the CHMv2 head.
        norm_strategy (str): Choices are 'linear', 'softmax', 'sigmoid' or 'chmv2_mixlog', for the conversion of features to depth logits

        Example:
        x = depth_model(input_image)  # N C H W
        - If pure regression (C == 1), depth is obtained by scaling and/or shifting x
        - If C > 1, bins are used:
            Depth is obtained as a weighted sum of depth bins, where weights are predicted logits. (see AdaBins [1] for more details)

        [1] AdaBins: https://github.com/shariqfarooq123/AdaBins
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert bins_strategy in ["linear", "log", "chmv2_mixlog"], "Support bins_strategy: linear, log, chmv2_mixlog"
        assert norm_strategy in ["linear", "softmax", "sigmoid", "chmv2_mixlog"], (
            "Support norm_strategy: linear, softmax, sigmoid, chmv2_mixlog"
        )

        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

    def forward(self, x):
        n_bins = x.shape[1]  # N n_bins H W
        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth)),
                    torch.log(torch.tensor(self.max_depth)),
                    n_bins,
                    device=x.device,
                )
                bins = torch.exp(bins)
            else:  # bins_strategy = "chmv2_mixlog"
                bins = create_chmv2_mixlog_bins(self.min_depth, self.max_depth, n_bins, x.device)

            # following Adabins, default is linear
            if self.norm_strategy in ["linear", "softmax", "sigmoid"]:
                if self.norm_strategy == "linear":
                    logit = torch.relu(x)
                    eps = 0.1
                    logit = logit + eps
                    logit = logit / logit.sum(dim=1, keepdim=True)
                elif self.norm_strategy == "softmax":
                    logit = torch.softmax(x, dim=1)
                else:  # norm_strategy = "sigmoid"
                    logit = torch.sigmoid(x)
                    logit = logit / logit.sum(dim=1, keepdim=True)
                output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)

            else:  # norm_strategy = "chmv2_mixlog"
                output = create_outputs_with_chmv2_mixlog_norm(x, bins)
        else:
            # standard regression
            output = torch.relu(x) + self.min_depth
        return output

