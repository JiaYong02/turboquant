"""TurboQuant_prod (Algorithm 2) — inner-product-optimal quantizer.

Two-stage quantizer: (b-1) bits MSE quantization + 1-bit QJL on the residual.
Provides unbiased inner product estimation: E[<y, x_hat>] = <y, x>.
"""

import numpy as np

from .qjl import QJL
from .quantizer_mse import TurboQuantMSE


class TurboQuantProd:
    """Inner-product-optimal quantizer from Algorithm 2 of the TurboQuant paper.

    Uses (b-1) bits for MSE stage and 1 bit (QJL) for residual correction.
    For b=1, this is pure QJL with no MSE stage.

    Args:
        d: Vector dimension.
        b: Total bits per coordinate (>= 1).
        seed: Random seed. MSE stage uses `seed`, QJL uses `seed + 1`.
    """

    def __init__(self, d: int, b: int, seed: int | None = None):
        self.d = d
        self.b = b

        if b > 1:
            self.mse = TurboQuantMSE(d, b - 1, seed=seed)
        else:
            self.mse = None

        self.qjl = QJL(d, seed=(seed + 1 if seed is not None else None))

    def quantize(self, x: np.ndarray) -> dict:
        """Quantize unit vector(s) using Algorithm 2.

        Args:
            x: Input of shape (d,) or (n, d), assumed unit norm.

        Returns:
            Dict with 'mse' (MSE indices or None), 'qjl' (sign bits),
            and 'gamma' (residual norms).
        """
        if self.mse is not None:
            mse_q = self.mse.quantize(x)
            x_mse = self.mse.dequantize(mse_q)
        else:
            mse_q = None
            x_mse = np.zeros_like(x)

        r = x - x_mse
        gamma = np.linalg.norm(r, axis=-1)
        qjl_signs = self.qjl.quantize(r)

        return {"mse": mse_q, "qjl": qjl_signs, "gamma": gamma}

    def dequantize(self, quantized: dict) -> np.ndarray:
        """Reconstruct vector(s) from quantized representation.

        Args:
            quantized: Dict from quantize().

        Returns:
            Reconstructed vector(s), same shape as original input.
        """
        if quantized["mse"] is not None:
            x_mse = self.mse.dequantize(quantized["mse"])
        else:
            x_mse = 0.0

        x_qjl = self.qjl.dequantize(quantized["qjl"], quantized["gamma"])
        return x_mse + x_qjl
