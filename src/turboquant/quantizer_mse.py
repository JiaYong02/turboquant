"""TurboQuant_mse (Algorithm 1) — MSE-optimal vector quantizer.

Applies a random orthogonal rotation then scalar Lloyd-Max quantization
on each coordinate independently.
"""

import numpy as np

from .codebook import LloydMaxCodebook
from .rotation import RandomRotation


class TurboQuantMSE:
    """MSE-optimal quantizer from Algorithm 1 of the TurboQuant paper.

    Pipeline: x -> rotate -> scalar quantize each coordinate -> dequantize -> unrotate

    Args:
        d: Vector dimension.
        b: Bits per coordinate.
        seed: Random seed for the rotation matrix.
    """

    def __init__(self, d: int, b: int, seed: int | None = None):
        self.d = d
        self.b = b
        self.rotation = RandomRotation(d, seed=seed)
        self.codebook = LloydMaxCodebook(d, b)

    def quantize(self, x: np.ndarray) -> dict:
        """Quantize unit vector(s).

        Args:
            x: Input of shape (d,) or (n, d), assumed unit norm.

        Returns:
            Dict with 'idx': centroid indices, dtype uint8.
        """
        y = self.rotation.rotate(x)
        idx = self.codebook.quantize(y)
        return {"idx": idx}

    def dequantize(self, quantized: dict) -> np.ndarray:
        """Reconstruct vector(s) from quantized representation.

        Args:
            quantized: Dict with 'idx' from quantize().

        Returns:
            Reconstructed vector(s), same shape as original input.
        """
        y_tilde = self.codebook.dequantize(quantized["idx"])
        return self.rotation.unrotate(y_tilde)

    def quantize_with_norm(self, x: np.ndarray) -> tuple[dict, np.ndarray]:
        """Quantize non-unit vector(s) by normalizing first.

        Args:
            x: Input of shape (d,) or (n, d).

        Returns:
            Tuple of (quantized dict, norms). Norms is scalar for single
            vector or array of shape (n,) for batch.
        """
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            x_unit = x / norm if norm > 0 else x
            return self.quantize(x_unit), norm
        else:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x_unit = np.where(norms > 0, x / norms, x)
            return self.quantize(x_unit), norms.squeeze()

    def dequantize_with_norm(
        self, quantized: dict, norms: np.ndarray | float
    ) -> np.ndarray:
        """Dequantize and rescale by stored norms.

        Args:
            quantized: Dict from quantize_with_norm().
            norms: Scalar or array of shape (n,).

        Returns:
            Reconstructed vector(s) at original scale.
        """
        x_unit = self.dequantize(quantized)
        if x_unit.ndim == 1:
            return x_unit * norms
        else:
            return x_unit * np.asarray(norms)[:, np.newaxis]
