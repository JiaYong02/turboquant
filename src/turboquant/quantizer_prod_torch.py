"""PyTorch TurboQuant_prod (Algorithm 2) — inner-product-optimal quantizer.

Port of quantizer_prod.py — (b-1) bits MSE + 1-bit QJL on residual.
"""

import torch

from .qjl_torch import QJLTorch
from .quantizer_mse_torch import TurboQuantMSETorch


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
