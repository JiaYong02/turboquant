"""PyTorch TurboQuant_mse (Algorithm 1) — MSE-optimal vector quantizer.

Port of quantizer_mse.py — random rotation + scalar Lloyd-Max quantization.
"""

import torch

from .codebook_torch import LloydMaxCodebookTorch
from .rotation_torch import RandomRotationTorch


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

        Args:
            x: Tensor of shape (d,) or (n, d).

        Returns:
            Tuple of (quantized dict, norms).
        """
        if x.dim() == 1:
            norm = torch.linalg.norm(x)
            x_unit = torch.where(norm > 0, x / norm, x)
            return self.quantize(x_unit), norm
        else:
            norms = torch.linalg.norm(x, dim=1, keepdim=True)
            x_unit = torch.where(norms > 0, x / norms, x)
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
