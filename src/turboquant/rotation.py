"""Random orthogonal rotation matrix for TurboQuant.

Generates a Haar-distributed random rotation (uniform over the orthogonal group)
by QR-decomposing a random Gaussian matrix.
"""

import numpy as np


class RandomRotation:
    """A random d x d orthogonal rotation matrix.

    After rotation, each coordinate of a unit vector follows a Beta distribution
    (Lemma 1 of TurboQuant paper), enabling optimal scalar quantization.

    Args:
        d: Dimension of the rotation matrix.
        seed: Random seed for reproducibility.
    """

    def __init__(self, d: int, seed: int | None = None):
        self.d = d
        self.seed = seed
        self.Pi = self._generate(d, seed)

    def _generate(self, d: int, seed: int | None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        A = rng.randn(d, d)
        Q, R = np.linalg.qr(A)
        # Sign correction: ensure Haar-distributed rotation (not reflection).
        # Multiply columns of Q by sign of diagonal of R.
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs[np.newaxis, :]
        return Q

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation: y = Pi @ x.

        Args:
            x: Vector of shape (d,) or batch of shape (n, d).

        Returns:
            Rotated vector(s), same shape as input.
        """
        return x @ self.Pi.T

    def unrotate(self, y: np.ndarray) -> np.ndarray:
        """Apply inverse rotation: x = Pi^T @ y.

        Since Pi is orthogonal, Pi^{-1} = Pi^T.

        Args:
            y: Rotated vector of shape (d,) or batch of shape (n, d).

        Returns:
            Original-space vector(s), same shape as input.
        """
        return y @ self.Pi
