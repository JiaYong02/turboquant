"""PyTorch Lloyd-Max codebook wrapper for TurboQuant.

Reuses the NumPy LloydMaxCodebook for scipy-based computation, then stores
centroids and boundaries as torch tensors on the target device.
"""

import torch

from .codebook import LloydMaxCodebook


class LloydMaxCodebookTorch:
    """Lloyd-Max codebook with PyTorch tensor storage.

    Codebook computation is delegated to the NumPy LloydMaxCodebook (uses scipy).
    Results are converted to torch.float32 tensors on the specified device.

    Args:
        d: Dimension of the ambient space.
        b: Bit-width (2^b centroids).
        device: Target device for tensors.
        use_gaussian_approx: If True, use N(0, 1/d) approximation.
    """

    def __init__(
        self,
        d: int,
        b: int,
        device: torch.device | str | None = None,
        use_gaussian_approx: bool | None = None,
    ):
        self._d = d
        self._b = b
        self._n_centroids = 2**b

        np_codebook = LloydMaxCodebook(d, b, use_gaussian_approx=use_gaussian_approx)
        self._centroids = torch.tensor(
            np_codebook.centroids, dtype=torch.float32, device=device
        )
        self._boundaries = torch.tensor(
            np_codebook.boundaries, dtype=torch.float32, device=device
        )
        self._mse_cost = np_codebook.mse_cost

    @property
    def centroids(self) -> torch.Tensor:
        """Sorted tensor of 2^b centroid values."""
        return self._centroids

    @property
    def boundaries(self) -> torch.Tensor:
        """Sorted tensor of 2^b + 1 boundary values."""
        return self._boundaries

    @property
    def mse_cost(self) -> float:
        """Per-coordinate MSE cost C(f_X, b)."""
        return self._mse_cost

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        """Map scalar values to nearest centroid indices.

        Args:
            values: Tensor of scalar values (any shape).

        Returns:
            Tensor of uint8 indices, same shape as values.
        """
        interior = self._boundaries[1:-1]
        return torch.searchsorted(interior, values).to(torch.uint8)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Map centroid indices back to centroid values.

        Args:
            indices: Tensor of integer indices (any shape).

        Returns:
            Tensor of centroid values, same shape as indices.
        """
        return self._centroids[indices.long()]
