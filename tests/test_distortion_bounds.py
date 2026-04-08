"""Distortion bound validation — reproduces paper's theoretical guarantees.

Verifies that empirical MSE and inner-product distortions fall within
the proven upper and lower bounds from Theorems 2 and 3.
"""

import numpy as np
import pytest

from turboquant.quantizer_mse import TurboQuantMSE
from turboquant.quantizer_prod import TurboQuantProd
from turboquant.utils import random_unit_vectors


D = 512
N_VECTORS = 1000


class TestMSEDistortionBounds:
    """D_mse should satisfy: 1/4^b <= D_mse <= sqrt(3*pi)/2 * 1/4^b."""

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_mse_within_bounds(self, b):
        rng = np.random.RandomState(42)
        x = random_unit_vectors(N_VECTORS, D, rng)

        q = TurboQuantMSE(d=D, b=b, seed=0)
        x_hat = q.dequantize(q.quantize(x))

        d_mse = np.mean(np.sum((x - x_hat) ** 2, axis=1))

        lower = 1.0 / (4**b)
        upper = np.sqrt(3) * np.pi / 2 * (1.0 / (4**b))

        # Allow 20% margin for statistical noise
        assert d_mse >= lower * 0.8, (
            f"b={b}: D_mse={d_mse:.6f} below lower bound {lower:.6f}"
        )
        assert d_mse <= upper * 1.2, (
            f"b={b}: D_mse={d_mse:.6f} above upper bound {upper:.6f}"
        )


class TestProdDistortionBounds:
    """D_prod should satisfy bounds from Theorem 3.

    D_prod = E[(y^T x_hat - y^T x)^2] for fixed unit x, y.
    Expectation is over the quantizer's random matrices.
    """

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_prod_within_bounds(self, b):
        n_seeds = 300
        rng = np.random.RandomState(42)

        x = random_unit_vectors(1, D, rng)[0]
        y = random_unit_vectors(1, D, rng)[0]
        true_ip = x @ y

        errors_sq = np.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProd(d=D, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            errors_sq[seed] = (y @ x_hat - true_ip) ** 2

        d_prod = np.mean(errors_sq)

        # For unit y: ||y||^2 = 1
        lower = 1.0 / (D * 4**b)
        upper = np.sqrt(3) * np.pi**2 / (2 * D) * (1.0 / 4**b)

        # Allow generous margin for statistical noise
        assert d_prod >= lower * 0.2, (
            f"b={b}: D_prod={d_prod:.8f} below lower bound {lower:.8f}"
        )
        assert d_prod <= upper * 2.0, (
            f"b={b}: D_prod={d_prod:.8f} above upper bound {upper:.8f}"
        )


class TestProdUnbiasedness:
    """Inner product estimation should be unbiased: mean error ≈ 0."""

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_mean_error_centered_at_zero(self, b):
        n_seeds = 300
        rng = np.random.RandomState(42)

        x = random_unit_vectors(1, D, rng)[0]
        y = random_unit_vectors(1, D, rng)[0]
        true_ip = x @ y

        errors = np.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProd(d=D, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            errors[seed] = y @ x_hat - true_ip

        mean_error = np.mean(errors)
        np.testing.assert_allclose(mean_error, 0.0, atol=0.02)


class TestDistortionDecreasing:
    """More bits should give lower distortion."""

    def test_mse_distortion_decreases_with_bits(self):
        rng = np.random.RandomState(42)
        x = random_unit_vectors(500, D, rng)

        distortions = []
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(d=D, b=b, seed=0)
            x_hat = q.dequantize(q.quantize(x))
            mse = np.mean(np.sum((x - x_hat) ** 2, axis=1))
            distortions.append(mse)

        for i in range(len(distortions) - 1):
            assert distortions[i + 1] < distortions[i], (
                f"D_mse did not decrease: b={i+1}={distortions[i]:.4f}, "
                f"b={i+2}={distortions[i+1]:.4f}"
            )

    def test_prod_distortion_decreases_with_bits(self):
        n_vectors = 500

        distortions = []
        for b in [1, 2, 3]:
            q = TurboQuantProd(d=D, b=b, seed=0)
            rng = np.random.RandomState(42)
            x = random_unit_vectors(n_vectors, D, rng)
            y = random_unit_vectors(n_vectors, D, np.random.RandomState(99))

            x_hat = q.dequantize(q.quantize(x))
            estimated_ips = np.sum(y * x_hat, axis=1)
            true_ips = np.sum(y * x, axis=1)
            distortions.append(np.mean((estimated_ips - true_ips) ** 2))

        for i in range(len(distortions) - 1):
            assert distortions[i + 1] < distortions[i]
