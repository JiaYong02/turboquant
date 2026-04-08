"""Math helpers for TurboQuant: Beta PDF, Gaussian approximation, vector utilities."""

import numpy as np
from scipy.special import gammaln


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Exact PDF for a coordinate of a random point on the unit hypersphere S^{d-1}.

    From Lemma 1 of the paper:
        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    for x in [-1, 1].

    Uses log-gamma to avoid overflow at large d.

    Args:
        x: Scalar or array of values in [-1, 1].
        d: Dimension of the ambient space (must be >= 2).

    Returns:
        PDF values, same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    log_coeff = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    exponent = (d - 3) / 2

    mask = np.abs(x) < 1.0
    result = np.zeros_like(x)
    result[mask] = np.exp(log_coeff + exponent * np.log(1 - x[mask] ** 2))
    return result


def gaussian_approx_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Gaussian N(0, 1/d) approximation of the Beta PDF for large d.

    Accurate for d >= 50.

    Args:
        x: Scalar or array of values.
        d: Dimension.

    Returns:
        PDF values, same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(d / (2 * np.pi)) * np.exp(-d * x**2 / 2)


def random_unit_vectors(n: int, d: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate n random unit vectors uniformly distributed on S^{d-1}.

    Args:
        n: Number of vectors.
        d: Dimension.
        rng: Random state for reproducibility.

    Returns:
        Array of shape (n, d) with unit-norm rows.
    """
    if rng is None:
        rng = np.random.RandomState()
    x = rng.randn(n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms
