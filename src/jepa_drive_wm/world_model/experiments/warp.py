"""
Forward geometric transport ("project & splat") of grid-shaped values.

Given per-cell values living on a source camera's pixel grid (RGB, V-JEPA
tokens, ...), per-cell metric depth, pinhole intrinsics, and a rigid transform
to a destination camera, transport the values by:

    1. back-projecting every valid source cell to a 3D point,
    2. mapping the points into the destination camera,
    3. projecting onto the destination grid,
    4. keeping the nearest point per destination cell (z-buffer).

The same code path serves full-resolution RGB and the 24x78 V-JEPA patch
lattice; the caller just supplies intrinsics that match the grid (see
scale_intrinsics / patch_intrinsics).

Conventions (matching lift_splat and KITTISequence):
    * camera axes: x = right, y = down, z = forward
    * integer grid coordinates are cell centres; K uses the same convention
    * T_dst_src maps points from source into destination camera coordinates,
      p_dst = T_dst_src @ p_src — exactly what get_camera_se3(i, j) returns
      for warping frame i into frame j.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def scale_intrinsics(K: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
    """Intrinsics of the same camera after resampling the grid by (scale_x, scale_y).

    With integer coordinates at cell centres, resizing an (H, W) grid to
    (H * scale_y, W * scale_x) maps u -> (u + 0.5) * scale_x - 0.5. Folding that
    affine map into K gives the resized grid's intrinsics. The half-cell term is
    what makes the 1/16 patchification correct: the naive cx/16 is off by ~half
    a patch on the V-JEPA lattice.
    """
    K = torch.as_tensor(K, dtype=torch.float64).clone()
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] = (K[0, 2] + 0.5) * scale_x - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * scale_y - 0.5
    return K


def patch_intrinsics(K_image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Intrinsics of the token lattice: one 'pixel' per patch_size x patch_size patch."""
    return scale_intrinsics(K_image, 1.0 / patch_size, 1.0 / patch_size)


def forward_splat(
    values: torch.Tensor,
    depth: torch.Tensor,
    K_src: torch.Tensor,
    T_dst_src: torch.Tensor,
    K_dst: torch.Tensor | None = None,
    out_hw: tuple[int, int] | None = None,
    src_valid: torch.Tensor | None = None,
    min_depth: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-buffered nearest-cell forward warp of `values` into the destination view.

    values:    (H, W, C) per-cell values to transport
    depth:     (H, W) metric depth of each source cell
    K_src:     (3, 3) intrinsics of the source grid
    T_dst_src: (4, 4) rigid transform, p_dst = T_dst_src @ p_src
    K_dst:     (3, 3) intrinsics of the destination grid (default: K_src)
    out_hw:    destination grid shape (default: source shape)
    src_valid: (H, W) optional bool mask of source cells to transport

    Returns (warped, valid, depth_dst):
        warped:    (h, w, C) transported values, zero where nothing landed
        valid:     (h, w) bool, True where at least one source cell landed
        depth_dst: (h, w) destination-frame depth of the winning cell, 0 where empty

    Occlusions are handled by the z-buffer (nearest point wins); disocclusions
    show up as valid == False holes.
    """
    values = torch.as_tensor(values)
    depth = torch.as_tensor(depth, dtype=torch.float32)
    H, W = depth.shape
    if values.shape[:2] != (H, W):
        raise ValueError(f"values {tuple(values.shape)} does not match depth {(H, W)}")
    if K_dst is None:
        K_dst = K_src
    out_h, out_w = out_hw if out_hw is not None else (H, W)

    K_src = torch.as_tensor(K_src, dtype=torch.float32)
    K_dst = torch.as_tensor(K_dst, dtype=torch.float32)
    T = torch.as_tensor(T_dst_src, dtype=torch.float32)

    C = values.shape[-1] if values.ndim == 3 else 1
    flat_values = values.reshape(H * W, C)

    keep_src = torch.isfinite(depth) & (depth > 0)
    if src_valid is not None:
        keep_src = keep_src & src_valid.bool()

    empty_warped = values.new_zeros(out_h, out_w, C)
    empty_valid = torch.zeros(out_h, out_w, dtype=torch.bool)
    empty_depth = torch.zeros(out_h, out_w)
    if not keep_src.any():
        return empty_warped, empty_valid, empty_depth

    vs, us = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    us, vs = us[keep_src], vs[keep_src]
    z = depth[keep_src]
    vals = flat_values[keep_src.reshape(-1)]

    # Back-project to source camera coordinates.
    x = (us - K_src[0, 2]) * z / K_src[0, 0]
    y = (vs - K_src[1, 2]) * z / K_src[1, 1]
    points = torch.stack([x, y, z], dim=-1)

    # Into the destination camera.
    points = points @ T[:3, :3].T + T[:3, 3]
    z_dst = points[:, 2]

    keep = z_dst > min_depth
    u_dst = torch.round(K_dst[0, 0] * points[:, 0] / z_dst + K_dst[0, 2]).long()
    v_dst = torch.round(K_dst[1, 1] * points[:, 1] / z_dst + K_dst[1, 2]).long()
    keep &= (u_dst >= 0) & (u_dst < out_w) & (v_dst >= 0) & (v_dst < out_h)
    if not keep.any():
        return empty_warped, empty_valid, empty_depth

    u_dst, v_dst, z_dst, vals = u_dst[keep], v_dst[keep], z_dst[keep], vals[keep]

    # Z-buffer: sort by (cell, depth) so the first entry of each cell group is
    # its nearest point (same sort-key idiom as LiftSplat._splat_top).
    cells = v_dst * out_w + u_dst
    big = float(z_dst.max()) + 1.0
    order = torch.argsort(cells.double() * big + z_dst.double())
    cells_sorted = cells[order]
    is_first = torch.ones_like(cells_sorted, dtype=torch.bool)
    is_first[1:] = cells_sorted[1:] != cells_sorted[:-1]
    winners = order[is_first]
    win_cells = cells_sorted[is_first]

    warped = vals.new_zeros(out_h * out_w, C)
    warped[win_cells] = vals[winners]
    valid = torch.zeros(out_h * out_w, dtype=torch.bool)
    valid[win_cells] = True
    depth_dst = torch.zeros(out_h * out_w)
    depth_dst[win_cells] = z_dst[winners]

    warped = warped.reshape(out_h, out_w, C)
    if values.ndim == 2:
        warped = warped.squeeze(-1)
    return warped, valid.reshape(out_h, out_w), depth_dst.reshape(out_h, out_w)


def forward_splat_soft(
    values: torch.Tensor,
    depth: torch.Tensor,
    K_src: torch.Tensor,
    T_dst_src: torch.Tensor,
    K_dst: torch.Tensor | None = None,
    out_hw: tuple[int, int] | None = None,
    src_valid: torch.Tensor | None = None,
    min_depth: float = 1e-3,
    occlusion_beta: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft forward warp: bilinear footprint + depth-softmax occlusion.

    forward_splat's one-cell z-buffer aliases on coarse lattices: under forward
    motion the scene expands, source cells spread apart, and regular holes
    appear between winners. Here each source cell instead spreads its value
    over the 4 destination cells around its continuous projection, weighted by
    bilinear overlap times exp(-occlusion_beta * (z - z_min_cell)), so nearer
    surfaces still dominate (soft z-buffer) but coverage stays dense.

    Same signature and returns as forward_splat.
    """
    values = torch.as_tensor(values)
    depth = torch.as_tensor(depth, dtype=torch.float32)
    H, W = depth.shape
    if K_dst is None:
        K_dst = K_src
    out_h, out_w = out_hw if out_hw is not None else (H, W)

    K_src = torch.as_tensor(K_src, dtype=torch.float32)
    K_dst = torch.as_tensor(K_dst, dtype=torch.float32)
    T = torch.as_tensor(T_dst_src, dtype=torch.float32)

    C = values.shape[-1] if values.ndim == 3 else 1
    flat_values = values.reshape(H * W, C)

    keep_src = torch.isfinite(depth) & (depth > 0)
    if src_valid is not None:
        keep_src = keep_src & src_valid.bool()

    empty = (
        values.new_zeros(out_h, out_w, C),
        torch.zeros(out_h, out_w, dtype=torch.bool),
        torch.zeros(out_h, out_w),
    )
    if not keep_src.any():
        return empty

    vs, us = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    us, vs = us[keep_src], vs[keep_src]
    z = depth[keep_src]
    vals = flat_values[keep_src.reshape(-1)].float()

    x = (us - K_src[0, 2]) * z / K_src[0, 0]
    y = (vs - K_src[1, 2]) * z / K_src[1, 1]
    points = torch.stack([x, y, z], dim=-1) @ T[:3, :3].T + T[:3, 3]
    z_dst = points[:, 2]

    keep = z_dst > min_depth
    if not keep.any():
        return empty
    points, z_dst, vals = points[keep], z_dst[keep], vals[keep]
    u = K_dst[0, 0] * points[:, 0] / z_dst + K_dst[0, 2]
    v = K_dst[1, 1] * points[:, 1] / z_dst + K_dst[1, 2]

    # The 4 destination cells under each continuous projection. Contributions
    # with negligible footprint are dropped BEFORE occlusion weighting — else a
    # nearer surface grazing a cell with ~zero bilinear overlap would still win
    # the depth softmax and overwrite a fully-overlapping farther one.
    min_footprint = 1e-3
    u0, v0 = torch.floor(u), torch.floor(v)
    corners = []
    for du in (0.0, 1.0):
        for dv in (0.0, 1.0):
            uc, vc = u0 + du, v0 + dv
            w_bilinear = (1.0 - (u - uc).abs()) * (1.0 - (v - vc).abs())
            inside = (
                (uc >= 0) & (uc < out_w) & (vc >= 0) & (vc < out_h)
                & (w_bilinear > min_footprint)
            )
            cells = (vc * out_w + uc).long()[inside]
            corners.append((cells, w_bilinear[inside], inside))

    # Pass 1: nearest depth reaching each cell (for the occlusion weighting).
    n_cells = out_h * out_w
    z_min = torch.full((n_cells,), torch.inf)
    for cells, _, inside in corners:
        z_min.scatter_reduce_(0, cells, z_dst[inside], reduce="amin")

    # Pass 2: accumulate values with bilinear x occlusion weights.
    num = torch.zeros(n_cells, C)
    den = torch.zeros(n_cells)
    depth_num = torch.zeros(n_cells)
    for cells, w_bilinear, inside in corners:
        w = w_bilinear * torch.exp(-occlusion_beta * (z_dst[inside] - z_min[cells]))
        num.index_add_(0, cells, vals[inside] * w.unsqueeze(-1))
        den.index_add_(0, cells, w)
        depth_num.index_add_(0, cells, z_dst[inside] * w)

    valid = den > 1e-8
    den_safe = den.clamp(min=1e-8).unsqueeze(-1)
    warped = (num / den_safe).to(values.dtype)
    warped[~valid] = 0
    depth_dst = depth_num / den_safe.squeeze(-1)
    depth_dst[~valid] = 0

    warped = warped.reshape(out_h, out_w, C)
    if values.ndim == 2:
        warped = warped.squeeze(-1)
    return warped, valid.reshape(out_h, out_w), depth_dst.reshape(out_h, out_w)


def fill_holes(
    image: torch.Tensor,
    valid: torch.Tensor,
    iterations: int = 8,
    fill_value: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Fill invalid cells with the mean of their valid 3x3 neighbours, iteratively.

    Grows the valid region by one cell per iteration, so small cracks left by
    forward splatting close while large disocclusions keep their (visible)
    fill_value. Returns a new (H, W, C) tensor; `valid` is not modified.
    """
    x = image.permute(2, 0, 1).unsqueeze(0).float()          # (1, C, H, W)
    m = valid.reshape(1, 1, *valid.shape).float()
    for _ in range(iterations):
        if m.all():
            break
        num = F.avg_pool2d(x * m, 3, stride=1, padding=1)
        den = F.avg_pool2d(m, 3, stride=1, padding=1)
        neighbour_mean = num / den.clamp(min=1e-8)
        reachable = den > 0
        grow = (m == 0) & reachable
        x = torch.where(grow, neighbour_mean, x)
        m = torch.where(grow, torch.ones_like(m), m)

    fill = torch.as_tensor(fill_value, dtype=x.dtype).reshape(1, -1, 1, 1)
    x = torch.where(m == 0, fill.expand_as(x), x)
    return x.squeeze(0).permute(1, 2, 0)


def _self_test() -> None:
    """Cheap invariants, no dataset needed. Raises on failure."""
    torch.manual_seed(0)
    H, W = 48, 96
    K = torch.tensor([[80.0, 0, W / 2 - 0.5], [0, 80.0, H / 2 - 0.5], [0, 0, 1]])
    depth = 5.0 + 5.0 * torch.rand(H, W)
    values = torch.rand(H, W, 3)

    # Identity transform must reproduce the input exactly.
    warped, valid, depth_dst = forward_splat(values, depth, K, torch.eye(4))
    assert valid.all(), "identity warp left holes"
    assert torch.allclose(warped, values, atol=1e-5), "identity warp changed values"
    assert torch.allclose(depth_dst, depth, atol=1e-4), "identity warp changed depth"

    # Moving the camera forward by 2m must reduce destination depth by 2m.
    T = torch.eye(4)
    T[2, 3] = -2.0  # p_dst = p_src - 2 along z <=> camera moved +2m forward
    _, valid_fwd, depth_fwd = forward_splat(values, depth, K, T)
    err = (depth[valid_fwd] - depth_fwd[valid_fwd].add(2.0)).abs()
    # Winners come from the same or a nearer source cell, so compare medians.
    assert (depth_fwd[valid_fwd].median() - (depth.median() - 2.0)).abs() < 0.2

    # Patch intrinsics: projecting a random 3D point with K then converting the
    # pixel to patch coordinates must equal projecting with patch_intrinsics.
    p = 16
    Kp = patch_intrinsics(K, p)
    point = torch.tensor([1.3, -0.7, 9.0])
    u = K[0, 0] * point[0] / point[2] + K[0, 2]
    u_patch_via_pixel = (u - (p - 1) / 2) / p
    u_patch_direct = Kp[0, 0] * point[0] / point[2] + Kp[0, 2]
    assert abs(u_patch_via_pixel - u_patch_direct) < 1e-6, "patch intrinsics mismatch"

    # Soft splat under the identity lands each cell exactly on itself.
    warped_soft, valid_soft, _ = forward_splat_soft(values, depth, K, torch.eye(4))
    assert valid_soft.all(), "identity soft warp left holes"
    assert torch.allclose(warped_soft, values, atol=1e-4), "identity soft warp changed values"

    # Under forward motion the soft footprint must cover at least what the
    # nearest-cell z-buffer covers.
    _, valid_soft_fwd, _ = forward_splat_soft(values, depth, K, T)
    assert (valid_fwd & ~valid_soft_fwd).sum() == 0, "soft warp lost nearest-splat coverage"

    # Hole filling: a small hole gets neighbour values, a huge one the fill value.
    img = torch.ones(H, W, 3)
    hole_valid = torch.ones(H, W, dtype=torch.bool)
    hole_valid[10:12, 10:12] = False
    filled = fill_holes(img, hole_valid, iterations=4, fill_value=torch.zeros(3))
    assert torch.allclose(filled[10:12, 10:12], torch.ones(2, 2, 3)), "small hole not filled"

    print("warp.py self-test passed")


if __name__ == "__main__":
    _self_test()
