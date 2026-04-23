"""Outlier-channel calibration for TurboQuant split-bucket quantizers.

Phase B heuristic (Zandieh et al., §4.2): after rotation by ``Pi``, the
magnitude of rotated coordinates is highly non-uniform across channels. A
small fraction of "outlier" channels accounts for most of the energy.

Allocating a higher bit-width to those channels and a lower width to the
rest yields a better rate-distortion tradeoff than a single uniform width.
The selection is static per head — computed once at model load time from a
calibration batch — so the core data-oblivious property is preserved at
decode time (no per-prompt state).

This module computes the per-head permutation that places outlier
channels into the contiguous prefix ``[0, n_out)`` of the rotated axis
and folds it into the rotation matrix so the kernel sees a single
rotation with outliers already in place.
"""

from __future__ import annotations

import torch

from .quantizer_mse_torch import BatchedTurboQuantMSETorch
from .quantizer_prod_torch import BatchedTurboQuantProdTorch


def _rotate(Pi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched rotation ``y = x @ Pi^T`` with shape ``[H, N, D]``."""
    return torch.bmm(x, Pi.transpose(-1, -2))


def calibrate_outliers(
    quantizer: BatchedTurboQuantProdTorch | BatchedTurboQuantMSETorch,
    calib_batch: torch.Tensor,
    n_out: int,
) -> torch.Tensor:
    """Select the top-``n_out`` outlier channels per head from calibration data.

    For each head, computes per-channel RMS of the rotated calibration
    vectors ``Pi @ x`` and returns a permutation that places the
    highest-RMS ``n_out`` channels in the prefix ``[0, n_out)`` and the
    remainder in any stable order afterwards.

    Args:
        quantizer: A batched quantizer whose ``rotation.Pi`` has shape
            ``[H, D, D]``.
        calib_batch: Pre-rotation calibration vectors of shape
            ``[H, N, D]``. ``N`` should be large enough for a stable RMS
            estimate (a few hundred is typically sufficient).
        n_out: Number of outlier channels to select. Must satisfy
            ``0 <= n_out <= D``.

    Returns:
        Long tensor of shape ``[H, D]`` — a permutation where
        ``perm[h, :n_out]`` are outlier channels sorted by descending RMS
        and ``perm[h, n_out:]`` are the remaining channels sorted by
        ascending original index (stable order).
    """
    Pi = quantizer.rotation.Pi  # [H, D, D]
    H, D, _ = Pi.shape
    if n_out < 0 or n_out > D:
        raise ValueError(f"n_out must be in [0, D]; got n_out={n_out}, D={D}")
    if calib_batch.dim() != 3 or calib_batch.shape[0] != H or calib_batch.shape[-1] != D:
        raise ValueError(
            f"calib_batch must have shape [H={H}, N, D={D}]; "
            f"got {tuple(calib_batch.shape)}"
        )
    if calib_batch.shape[1] == 0:
        raise ValueError("calib_batch must contain at least one sample per head")

    x = calib_batch.to(device=Pi.device, dtype=torch.float32)
    y = _rotate(Pi, x)  # [H, N, D]
    rms = torch.sqrt(torch.mean(y.pow(2), dim=1))  # [H, D]

    # argsort by -rms places largest first; stable so ties break by original
    # channel index. This gives a deterministic perm that's invariant to
    # minor numerical noise in RMS for near-tied channels.
    perm = torch.argsort(-rms, dim=-1, stable=True)  # [H, D]
    return perm


def apply_outlier_permutation(
    quantizer: BatchedTurboQuantProdTorch | BatchedTurboQuantMSETorch,
    perm: torch.Tensor,
) -> None:
    """Fold an outlier permutation into the quantizer's rotation matrix.

    Rewrites ``quantizer.rotation.Pi`` so that ``y = x @ Pi_new^T`` has
    its rotated coordinates reordered by ``perm``: ``y_new[..., i] ==
    (old y)[..., perm[..., i]]``. After this call, outlier channels
    occupy the prefix ``[0, n_out)`` as required by the split-bucket
    kernel.

    The operation is idempotent-safe only relative to the permutation
    itself — calling it twice with the *same* perm re-applies it and
    will double-permute. Call at most once per quantizer.

    Args:
        quantizer: The quantizer whose rotation matrix is modified in place.
        perm: Long tensor of shape ``[H, D]`` from :func:`calibrate_outliers`.
    """
    Pi = quantizer.rotation.Pi  # [H, D, D]
    H, D, _ = Pi.shape
    if perm.shape != (H, D):
        raise ValueError(
            f"perm must have shape [{H}, {D}]; got {tuple(perm.shape)}"
        )
    idx = perm.to(device=Pi.device, dtype=torch.long).unsqueeze(-1).expand(-1, -1, D)
    quantizer.rotation.Pi = torch.gather(Pi, dim=1, index=idx).contiguous()
