"""Tests for TurboQuant_prod (Algorithm 2) — inner-product-optimal quantizer."""

import numpy as np
import pytest

from turboquant.quantizer_prod import TurboQuantProd
from turboquant.quantizer_mse import TurboQuantMSE
from turboquant.utils import random_unit_vectors


class TestShapes:
    """Shape preservation for single and batch inputs."""

    def test_single_vector(self):
        q = TurboQuantProd(d=128, b=2, seed=0)
        x = random_unit_vectors(1, 128, np.random.RandomState(0))[0]
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (128,)

    def test_batch(self):
        q = TurboQuantProd(d=128, b=2, seed=0)
        x = random_unit_vectors(20, 128, np.random.RandomState(0))
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (20, 128)


class TestB1SpecialCase:
    """b=1 should be pure QJL with no MSE component."""

    def test_no_mse_stage(self):
        q = TurboQuantProd(d=128, b=1, seed=0)
        assert q.mse is None

    def test_mse_field_is_none(self):
        q = TurboQuantProd(d=128, b=1, seed=0)
        x = random_unit_vectors(5, 128, np.random.RandomState(0))
        quantized = q.quantize(x)
        assert quantized["mse"] is None

    def test_b1_still_works(self):
        q = TurboQuantProd(d=128, b=1, seed=0)
        x = random_unit_vectors(10, 128, np.random.RandomState(0))
        x_hat = q.dequantize(q.quantize(x))
        assert x_hat.shape == x.shape


class TestUnbiasedness:
    """E[<y, x_hat>] ≈ <y, x> — the key property of TurboQuant_prod.

    Unbiasedness holds over the randomness of the projection matrices.
    We test with a fixed (x, y) pair across many quantizer seeds.
    """

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_inner_product_unbiased(self, b):
        d = 128
        n_seeds = 200
        rng = np.random.RandomState(42)

        x = random_unit_vectors(1, d, rng)[0]
        y = random_unit_vectors(1, d, rng)[0]
        true_ip = x @ y

        estimates = np.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProd(d=d, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            estimates[seed] = y @ x_hat

        mean_estimate = np.mean(estimates)
        np.testing.assert_allclose(mean_estimate, true_ip, atol=0.04)


class TestMSEIsBiased:
    """Control test: TurboQuantMSE IS biased for inner products."""

    def test_mse_quantizer_biased(self):
        """MSE quantizer at b=1 should show multiplicative bias ≈ 2/pi for inner products."""
        d = 128
        n_seeds = 200
        rng = np.random.RandomState(42)

        x = random_unit_vectors(1, d, rng)[0]
        y = random_unit_vectors(1, d, rng)[0]
        true_ip = x @ y

        estimates = np.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantMSE(d=d, b=1, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            estimates[seed] = y @ x_hat

        mean_estimate = np.mean(estimates)
        bias_ratio = mean_estimate / true_ip
        # MSE quantizer at b=1 has bias factor ≈ 2/pi ≈ 0.636
        assert 0.50 < bias_ratio < 0.80, f"Expected bias ~0.636, got ratio {bias_ratio}"


class TestInnerProductDistortion:
    """Variance of inner product estimates should match D_prod table values.

    D_prod = E[(y^T x_hat - y^T x)^2] for fixed unit x, y, varying the
    quantizer's random matrices. Paper: D_prod * d ≈ 1.57, 0.56, 0.18 for b=1,2,3.
    """

    @pytest.mark.parametrize(
        "b, expected_d_times_dprod",
        [(1, 1.57), (2, 0.56), (3, 0.18)],
    )
    def test_distortion_matches_paper(self, b, expected_d_times_dprod):
        d = 256
        n_seeds = 500
        rng = np.random.RandomState(42)

        x = random_unit_vectors(1, d, rng)[0]
        y = random_unit_vectors(1, d, rng)[0]
        true_ip = x @ y

        errors_sq = np.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProd(d=d, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            errors_sq[seed] = (y @ x_hat - true_ip) ** 2

        empirical_d_times_dprod = d * np.mean(errors_sq)
        np.testing.assert_allclose(
            empirical_d_times_dprod, expected_d_times_dprod, rtol=0.40
        )


class TestDifferentDimensions:
    """Works across various dimensions."""

    @pytest.mark.parametrize("d", [128, 256])
    def test_round_trip_various_d(self, d):
        rng = np.random.RandomState(42)
        x = random_unit_vectors(20, d, rng)
        q = TurboQuantProd(d=d, b=2, seed=0)
        x_hat = q.dequantize(q.quantize(x))
        assert x_hat.shape == x.shape
