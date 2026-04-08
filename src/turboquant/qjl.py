"""QJL (Quantized Johnson-Lindenstrauss) 1-bit quantizer.

Provides unbiased inner product estimation via random sign projections.
From Section 3 of the TurboQuant paper.
"""

import numpy as np


class QJL:
    """1-bit quantizer using random Gaussian projections.

    Quantizes a vector r to sign(S @ r) where S is a d x d random Gaussian matrix.
    Dequantization reconstructs via sqrt(pi/2)/d * gamma * S^T @ signs, which is
    an unbiased estimator of r when gamma = ||r||.

    Note:
        Stores a dense d x d projection matrix, requiring O(d^2) memory
        (e.g. ~8 MB for d=1024 in float64).

    Args:
        d: Dimension of the input vectors.
        seed: Random seed for reproducibility.
    """

    def __init__(self, d: int, seed: int | None = None):
        rng = np.random.RandomState(seed)
        self.S = rng.randn(d, d)
        self.d = d
        self._scale = np.sqrt(np.pi / 2) / d

    def quantize(self, r: np.ndarray) -> np.ndarray:
        """Compute sign(r @ S^T), mapping zeros to +1.

        Args:
            r: Input vector(s) of shape (d,) or (n, d).

        Returns:
            Sign array of shape matching input, dtype int8, values in {-1, +1}.
        """
        projected = r @ self.S.T
        signs = np.sign(projected).astype(np.int8)
        signs[signs == 0] = 1
        return signs

    def dequantize(self, signs: np.ndarray, gamma: float | np.ndarray) -> np.ndarray:
        """Reconstruct vector(s) from sign bits and norm.

        x_hat = sqrt(pi/2)/d * gamma * S^T @ signs

        Args:
            signs: Sign array of shape (d,) or (n, d), values in {-1, +1}.
            gamma: Scalar norm or array of norms of shape (n,).

        Returns:
            Reconstructed vector(s), same shape as signs.
        """
        signs_f = signs.astype(np.float64)
        if signs_f.ndim == 1:
            return self._scale * gamma * (self.S.T @ signs_f)
        else:
            gamma = np.asarray(gamma)
            return self._scale * gamma[:, np.newaxis] * (signs_f @ self.S)
