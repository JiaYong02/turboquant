"""PyTorch TurboQuant_prod (Algorithm 2) — inner-product-optimal quantizer.

Port of quantizer_prod.py — (b-1) bits MSE + 1-bit QJL on residual.
"""

import torch

from .qjl_torch import BatchedQJLTorch, QJLTorch
from .quantizer_mse_torch import BatchedTurboQuantMSETorch, TurboQuantMSETorch


class TurboQuantProdTorch:
    """Inner-product-optimal quantizer from Algorithm 2 (PyTorch).

    Uses (b-1) bits for MSE stage and 1 bit (QJL) for residual correction.
    For b=1, this is pure QJL with no MSE stage.

    Args:
        d: Vector dimension.
        b: Total bits per coordinate (>= 1).
        seed: Random seed. MSE stage uses `seed`, QJL uses `seed + 1`.
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

        if b > 1:
            self.mse = TurboQuantMSETorch(d, b - 1, seed=seed, device=device)
        else:
            self.mse = None

        self.qjl = QJLTorch(
            d, seed=(seed + 1 if seed is not None else None), device=device
        )

    def quantize(self, x: torch.Tensor) -> dict:
        """Quantize unit vector(s) using Algorithm 2.

        Args:
            x: Tensor of shape (d,) or (n, d), assumed unit norm.

        Returns:
            Dict with 'mse' (MSE indices or None), 'qjl' (sign bits),
            and 'gamma' (residual norms).
        """
        if self.mse is not None:
            mse_q = self.mse.quantize(x)
            x_mse = self.mse.dequantize(mse_q)
        else:
            mse_q = None
            x_mse = torch.zeros_like(x)

        r = x - x_mse
        gamma = torch.linalg.norm(r, dim=-1)
        qjl_signs = self.qjl.quantize(r)

        return {"mse": mse_q, "qjl": qjl_signs, "gamma": gamma}

    def dequantize(self, quantized: dict) -> torch.Tensor:
        """Reconstruct vector(s) from quantized representation.

        Args:
            quantized: Dict from quantize().

        Returns:
            Reconstructed tensor, same shape as original input.
        """
        if quantized["mse"] is not None:
            x_mse = self.mse.dequantize(quantized["mse"])
        else:
            x_mse = 0.0

        x_qjl = self.qjl.dequantize(quantized["qjl"], quantized["gamma"])
        return x_mse + x_qjl

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


class BatchedTurboQuantProdTorch:
    """Inner-product-optimal quantizer for H attention heads processed simultaneously.

    Uses (b-1) MSE bits + 1 QJL bit with stacked [H, D, D] matrices so all
    heads are quantized in a single batched GPU call.

    Args:
        d: Vector dimension.
        b: Total bits per coordinate (>= 1).
        num_heads: Number of heads (H).
        mse_seeds: List of H seeds for MSE rotation matrices.
        qjl_seeds: List of H seeds for QJL projection matrices.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        b: int,
        num_heads: int,
        mse_seeds: list[int],
        qjl_seeds: list[int],
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
        if n_out > 0 and (bits_hi < 2 or bits_lo < 2):
            # Prod needs at least one MSE bit after subtracting the QJL bit.
            raise ValueError(
                "bits_hi and bits_lo must both be >= 2 for Prod (one bit goes "
                f"to QJL); got bits_hi={bits_hi}, bits_lo={bits_lo}"
            )

        self.d = d
        self.b = b
        self.num_heads = num_heads
        self.n_out = n_out
        self.bits_hi = bits_hi
        self.bits_lo = bits_lo

        if n_out > 0:
            # K-side MSE takes (bucket_bits - 1); QJL contributes the final bit.
            self.mse: BatchedTurboQuantMSETorch | None = BatchedTurboQuantMSETorch(
                d,
                b=bits_hi - 1,  # single-bucket fallback param; unused when n_out>0
                num_heads=num_heads,
                seeds=mse_seeds,
                device=device,
                n_out=n_out,
                bits_hi=bits_hi - 1,
                bits_lo=bits_lo - 1,
            )
        elif b > 1:
            self.mse = BatchedTurboQuantMSETorch(
                d, b - 1, num_heads, mse_seeds, device
            )
        else:
            self.mse = None

        self.qjl = BatchedQJLTorch(d, num_heads, qjl_seeds, device)

    def quantize_with_norm(
        self, x: torch.Tensor
    ) -> tuple[dict, torch.Tensor]:
        """Quantize non-unit vectors for all heads using Algorithm 2.

        Internally operates in float32 regardless of input dtype.

        Args:
            x: Tensor of shape (H, N, D). Any floating-point dtype.

        Returns:
            Tuple of (q_dict, norms of shape (H, N)).
            q_dict keys: 'mse' (dict or None), 'qjl' (int8 [H,N,D]),
            'gamma' (float32 [H,N]).
        """
        x_f32 = x.float()
        norms = torch.linalg.norm(x_f32, dim=-1, keepdim=True)  # [H, N, 1]
        x_unit = torch.where(norms > 0, x_f32 / norms, x_f32)
        norms = norms.squeeze(-1)                                # [H, N]
        return self._quantize(x_unit), norms

    def _quantize(self, x: torch.Tensor) -> dict:
        """Quantize unit vectors for all heads (internal)."""
        if self.mse is not None:
            mse_q = self.mse.quantize(x)
            x_mse = self.mse.dequantize(mse_q)
        else:
            mse_q = None
            x_mse = torch.zeros_like(x)

        r = x - x_mse
        gamma = torch.linalg.norm(r, dim=-1)       # [H, N]
        qjl_signs = self.qjl.quantize(r)           # [H, N, D] int8

        return {"mse": mse_q, "qjl": qjl_signs, "gamma": gamma}

    def dequantize_with_norm(
        self, q_dict: dict, norms: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize and rescale for all heads.

        Args:
            q_dict: Dict from quantize_with_norm().
            norms: float32 tensor of shape (H, N).

        Returns:
            float32 tensor of shape (H, N, D).
        """
        if q_dict["mse"] is not None:
            x_mse = self.mse.dequantize(q_dict["mse"])
        else:
            x_mse = 0.0

        x_qjl = self.qjl.dequantize(q_dict["qjl"], q_dict["gamma"])
        x_unit = x_mse + x_qjl                    # [H, N, D]
        return x_unit * norms.unsqueeze(-1)        # [H, N, D]
