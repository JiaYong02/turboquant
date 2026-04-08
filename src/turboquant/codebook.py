"""Lloyd-Max optimal scalar quantizer for the Beta distribution on the unit hypersphere."""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from .utils import beta_pdf, gaussian_approx_pdf


class LloydMaxCodebook:
    """Compute and store an optimal Lloyd-Max codebook for a given dimension and bit-width.

    The codebook minimizes MSE for scalar quantization of coordinates of a random
    point on the unit hypersphere S^{d-1}, whose marginal distribution is the Beta
    PDF from Lemma 1 of the TurboQuant paper.

    Results are cached: repeated construction with the same (d, b, use_gaussian_approx)
    reuses the previously computed codebook.

    Args:
        d: Dimension of the ambient space (>= 2).
        b: Bit-width (number of bits per coordinate). Codebook has 2^b centroids.
        use_gaussian_approx: If True, use N(0, 1/d) approximation instead of exact
            Beta PDF. Default is None (auto: use Gaussian for d >= 50).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on centroid movement.
    """

    _cache: dict[tuple[int, int, bool], tuple[np.ndarray, np.ndarray, float]] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Remove all cached codebooks to reclaim memory."""
        cls._cache.clear()

    def __init__(
        self,
        d: int,
        b: int,
        use_gaussian_approx: bool | None = None,
        max_iter: int = 1000,
        tol: float = 1e-12,
    ):
        self._d = d
        self._b = b
        self._n_centroids = 2**b

        if use_gaussian_approx is None:
            use_gaussian_approx = d >= 50

        cache_key = (d, b, use_gaussian_approx)
        if cache_key in LloydMaxCodebook._cache:
            self._centroids, self._boundaries, self._mse_cost = LloydMaxCodebook._cache[cache_key]
            return

        if use_gaussian_approx:
            self._pdf = lambda x: gaussian_approx_pdf(x, d)
        else:
            self._pdf = lambda x: beta_pdf(x, d)

        self._centroids = self._compute_codebook(max_iter, tol)
        self._boundaries = self._compute_boundaries()
        self._mse_cost = self._compute_mse_cost()

        LloydMaxCodebook._cache[cache_key] = (
            self._centroids, self._boundaries, self._mse_cost
        )

    def _compute_codebook(self, max_iter: int, tol: float) -> np.ndarray:
        n = self._n_centroids
        pdf = self._pdf

        # Initialize from quantiles of N(0, 1/d)
        sigma = 1.0 / np.sqrt(self._d)
        quantiles = norm.ppf(
            np.linspace(0.5 / n, 1 - 0.5 / n, n), scale=sigma
        )
        # Clip to interior of [-1, 1] to avoid boundary singularities in the Beta PDF
        _CLIP_BOUND = 0.999
        centroids = np.clip(quantiles, -_CLIP_BOUND, _CLIP_BOUND)

        for _ in range(max_iter):
            # Voronoi boundaries = midpoints
            boundaries = np.empty(n + 1)
            boundaries[0] = -1.0
            boundaries[-1] = 1.0
            for i in range(n - 1):
                boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

            # Update centroids to conditional mean
            new_centroids = np.empty(n)
            for i in range(n):
                lo, hi = boundaries[i], boundaries[i + 1]
                if hi - lo < 1e-15:
                    new_centroids[i] = (lo + hi) / 2
                    continue
                numerator, _ = quad(lambda x: x * pdf(x), lo, hi)
                denominator, _ = quad(pdf, lo, hi)
                if denominator > 1e-15:
                    new_centroids[i] = numerator / denominator
                else:
                    new_centroids[i] = (lo + hi) / 2

            if np.max(np.abs(new_centroids - centroids)) < tol:
                centroids = new_centroids
                break
            centroids = new_centroids

        return np.sort(centroids)

    def _compute_boundaries(self) -> np.ndarray:
        n = self._n_centroids
        boundaries = np.empty(n + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n - 1):
            boundaries[i + 1] = (self._centroids[i] + self._centroids[i + 1]) / 2
        return boundaries

    def _compute_mse_cost(self) -> float:
        """Compute C(f_X, b) = integral of (x - Q(x))^2 * f(x) dx."""
        pdf = self._pdf
        total = 0.0
        for i in range(self._n_centroids):
            lo, hi = self._boundaries[i], self._boundaries[i + 1]
            c = self._centroids[i]
            cost, _ = quad(lambda x: (x - c) ** 2 * pdf(x), lo, hi)
            total += cost
        return total

    @property
    def centroids(self) -> np.ndarray:
        """Sorted array of 2^b centroid values."""
        return self._centroids

    @property
    def boundaries(self) -> np.ndarray:
        """Sorted array of 2^b + 1 boundary values (including -1 and 1)."""
        return self._boundaries

    @property
    def mse_cost(self) -> float:
        """Per-coordinate MSE cost C(f_X, b). Total MSE distortion ≈ d * C(f_X, b)."""
        return self._mse_cost

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Map scalar values to nearest centroid indices using boundary lookup.

        Args:
            values: Array of scalar values (any shape).

        Returns:
            Array of uint8 indices, same shape as values.
        """
        # Interior boundaries for searchsorted
        interior = self._boundaries[1:-1]
        indices = np.searchsorted(interior, values).astype(np.uint8)
        return indices

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Map centroid indices back to centroid values.

        Args:
            indices: Array of integer indices (any shape).

        Returns:
            Array of centroid values, same shape as indices.
        """
        return self._centroids[indices]
