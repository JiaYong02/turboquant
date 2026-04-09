"""PyTorch random orthogonal rotation matrix for TurboQuant.

Port of rotation.py — generates a Haar-distributed random rotation using
QR decomposition of a random Gaussian matrix, with GPU support.
"""

import torch


class RandomRotationTorch:
    """A random d x d orthogonal rotation matrix (PyTorch).

    Args:
        d: Dimension of the rotation matrix.
        seed: Random seed for reproducibility.
        device: Target device for the rotation matrix.
    """

    def __init__(
        self,
        d: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.d = d
        self.seed = seed
        self.Pi = self._generate(d, seed, device)

    def _generate(
        self, d: int, seed: int | None, device: torch.device | str | None
    ) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        A = torch.randn(d, d, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(A)
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.unsqueeze(0)
        if device is not None:
            Q = Q.to(device)
        return Q

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: y = Pi @ x.

        Args:
            x: Tensor of shape (d,) or (n, d).

        Returns:
            Rotated tensor, same shape as input.
        """
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: x = Pi^T @ y.

        Args:
            y: Rotated tensor of shape (d,) or (n, d).

        Returns:
            Original-space tensor, same shape as input.
        """
        return y @ self.Pi
