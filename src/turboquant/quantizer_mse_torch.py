"""PyTorch TurboQuant_mse (Algorithm 1) — MSE-optimal vector quantizer.

Port of quantizer_mse.py — random rotation + scalar Lloyd-Max quantization.
"""

import torch

from .codebook_torch import LloydMaxCodebookTorch
from .rotation_torch import BatchedRandomRotationTorch, RandomRotationTorch


class TurboQuantMSETorch:
    """MSE-optimal quantizer from Algorithm 1 (PyTorch).

    Pipeline: x -> rotate -> scalar quantize each coordinate -> dequantize -> unrotate

    Args:
        d: Vector dimension.
        b: Bits per coordinate.
        seed: Random seed for the rotation matrix.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        b: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.d = d
        self.b = b
        self.rotation = RandomRotationTorch(d, seed=seed, device=device)
        self.codebook = LloydMaxCodebookTorch(d, b, device=device)

    def quantize(self, x: torch.Tensor) -> dict:
        """Quantize unit vector(s).

        Args:
            x: Tensor of shape (d,) or (n, d), assumed unit norm.

        Returns:
            Dict with 'idx': centroid indices, dtype uint8.
        """
        y = self.rotation.rotate(x)
        idx = self.codebook.quantize(y)
        return {"idx": idx}

    def dequantize(self, quantized: dict) -> torch.Tensor:
        """Reconstruct vector(s) from quantized representation.

        Args:
            quantized: Dict with 'idx' from quantize().

        Returns:
            Reconstructed tensor, same shape as original input.
        """
        y_tilde = self.codebook.dequantize(quantized["idx"])
        return self.rotation.unrotate(y_tilde)

    def quantize_with_norm(
        self, x: torch.Tensor
    ) -> tuple[dict, torch.Tensor]:
        """Quantize non-unit vector(s) by normalizing first.

        Internally operates in float32 regardless of the input dtype, so
        callers running in fp16/bf16 do not need to cast before calling.
        Norms are always returned as float32.

        Args:
            x: Tensor of shape (d,) or (n, d). Any floating-point dtype.

        Returns:
            Tuple of (quantized dict, norms as float32).
        """
        x_f32 = x.float()  # internal math always fp32; caller's dtype is irrelevant
        if x_f32.dim() == 1:
            norm = torch.linalg.norm(x_f32)
            x_unit = torch.where(norm > 0, x_f32 / norm, x_f32)
            return self.quantize(x_unit), norm
        else:
            norms = torch.linalg.norm(x_f32, dim=1, keepdim=True)
            x_unit = torch.where(norms > 0, x_f32 / norms, x_f32)
            return self.quantize(x_unit), norms.squeeze(1)

    def dequantize_with_norm(
        self, quantized: dict, norms: torch.Tensor | float
    ) -> torch.Tensor:
        """Dequantize and rescale by stored norms.

        Args:
            quantized: Dict from quantize_with_norm().
            norms: Scalar or tensor of shape (n,).

        Returns:
            Reconstructed tensor at original scale.
        """
        x_unit = self.dequantize(quantized)
        if x_unit.dim() == 1:
            return x_unit * norms
        else:
            norms_t = torch.as_tensor(
                norms, dtype=torch.float32, device=x_unit.device
            )
            return x_unit * norms_t.unsqueeze(-1)


class BatchedTurboQuantMSETorch:
    """MSE-optimal quantizer for H attention heads processed simultaneously.

    Uses stacked [H, D, D] rotation matrices and a shared Lloyd-Max codebook
    to quantize all heads in a single batched GPU call.

    Optional outlier-channel split (Phase B): when `n_out > 0`, the rotated
    coordinates are split into an outlier prefix `[0, n_out)` quantized at
    `bits_hi` bits and a regular suffix `[n_out, D)` at `bits_lo` bits. The
    rotation is assumed to already place outliers in the prefix (calibration
    lives outside this class). `n_out == 0` is bit-identical to the old
    single-bucket behavior.

    Args:
        d: Vector dimension.
        b: Bits per coordinate (single-bucket path; ignored when n_out > 0).
        num_heads: Number of heads (H).
        seeds: List of H rotation seeds, one per head.
        device: Target device.
        n_out: Number of outlier channels in the rotated prefix. 0 disables
            the split.
        bits_hi: Bit-width for the outlier bucket. Required when n_out > 0.
        bits_lo: Bit-width for the regular bucket. Required when n_out > 0.
    """

    def __init__(
        self,
        d: int,
        b: int,
        num_heads: int,
        seeds: list[int],
        device: torch.device | str | None = None,
        *,
        n_out: int = 0,
        bits_hi: int | None = None,
        bits_lo: int | None = None,
    ):
        if n_out < 0 or n_out > d:
            raise ValueError(f"n_out must be in [0, d]; got n_out={n_out}, d={d}")
        if n_out > 0 and (bits_hi is None or bits_lo is None):
            raise ValueError("bits_hi and bits_lo are required when n_out > 0")

        self.d = d
        self.b = b
        self.num_heads = num_heads
        self.n_out = n_out
        self.bits_hi = bits_hi
        self.bits_lo = bits_lo
        self.rotation = BatchedRandomRotationTorch(d, num_heads, seeds, device)

        if n_out == 0:
            # Codebook is shared across heads (same distribution for all).
            self.codebook = LloydMaxCodebookTorch(d, b, device=device)
            self.codebook_hi: LloydMaxCodebookTorch | None = None
            self.codebook_lo: LloydMaxCodebookTorch | None = None
        else:
            # `d` drives the rotated-coord Beta(1/2, (D-1)/2) marginal, so it
            # stays at the full dimension for both buckets.
            self.codebook = None  # type: ignore[assignment]
            self.codebook_hi = LloydMaxCodebookTorch(d, bits_hi, device=device)
            self.codebook_lo = LloydMaxCodebookTorch(d, bits_lo, device=device)

    def quantize(self, x: torch.Tensor) -> dict:
        """Quantize unit vectors for all heads.

        Args:
            x: float32 tensor of shape (H, N, D), assumed unit norm.

        Returns:
            Single-bucket mode: dict with 'idx' uint8 [H, N, D].
            Split-bucket mode (n_out > 0): dict with 'idx_hi' uint8
            [H, N, n_out] and 'idx_lo' uint8 [H, N, D - n_out].
        """
        y = self.rotation.rotate(x)       # [H, N, D]
        if self.n_out == 0:
            idx = self.codebook.quantize(y)   # [H, N, D] uint8
            return {"idx": idx}
        idx_hi = self.codebook_hi.quantize(y[..., : self.n_out].contiguous())
        idx_lo = self.codebook_lo.quantize(y[..., self.n_out :].contiguous())
        return {"idx_hi": idx_hi, "idx_lo": idx_lo}

    def dequantize(self, quantized: dict) -> torch.Tensor:
        """Reconstruct unit vectors from indices for all heads.

        Args:
            quantized: Dict from quantize().

        Returns:
            float32 tensor of shape (H, N, D).
        """
        if self.n_out == 0:
            y_tilde = self.codebook.dequantize(quantized["idx"])  # [H, N, D]
        else:
            y_hi = self.codebook_hi.dequantize(quantized["idx_hi"])
            y_lo = self.codebook_lo.dequantize(quantized["idx_lo"])
            y_tilde = torch.cat([y_hi, y_lo], dim=-1)             # [H, N, D]
        return self.rotation.unrotate(y_tilde)                    # [H, N, D]

    def quantize_with_norm(
        self, x: torch.Tensor
    ) -> tuple[dict, torch.Tensor]:
        """Quantize non-unit vectors for all heads.

        Internally operates in float32 regardless of input dtype.

        Args:
            x: Tensor of shape (H, N, D). Any floating-point dtype.

        Returns:
            Tuple of (quantized dict, norms of shape (H, N)).
        """
        x_f32 = x.float()
        norms = torch.linalg.norm(x_f32, dim=-1, keepdim=True)  # [H, N, 1]
        x_unit = torch.where(norms > 0, x_f32 / norms, x_f32)
        return self.quantize(x_unit), norms.squeeze(-1)          # norms: [H, N]

    def dequantize_with_norm(
        self, quantized: dict, norms: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize and rescale by stored norms for all heads.

        Args:
            quantized: Dict from quantize_with_norm().
            norms: float32 tensor of shape (H, N).

        Returns:
            float32 tensor of shape (H, N, D).
        """
        x_unit = self.dequantize(quantized)    # [H, N, D]
        return x_unit * norms.unsqueeze(-1)    # [H, N, D]
