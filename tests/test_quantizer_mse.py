"""Tests for TurboQuant_mse (Algorithm 1) — MSE-optimal quantizer."""

import numpy as np
import pytest

from turboquant.quantizer_mse import TurboQuantMSE
from turboquant.utils import random_unit_vectors


class TestShapes:
    """Round-trip shape preservation."""

    def test_single_vector(self):
        q = TurboQuantMSE(d=128, b=2, seed=0)
        x = random_unit_vectors(1, 128, np.random.RandomState(0))[0]
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (128,)

    def test_batch(self):
        q = TurboQuantMSE(d=128, b=2, seed=0)
        x = random_unit_vectors(20, 128, np.random.RandomState(0))
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (20, 128)

    def test_idx_dtype_uint8(self):
        q = TurboQuantMSE(d=64, b=4, seed=0)
        x = random_unit_vectors(5, 64, np.random.RandomState(0))
        quantized = q.quantize(x)
        assert quantized["idx"].dtype == np.uint8


class TestDeterminism:
    """Same seed produces identical results."""

    def test_same_seed_same_output(self):
        x = random_unit_vectors(10, 128, np.random.RandomState(42))
        q1 = TurboQuantMSE(d=128, b=2, seed=7)
        q2 = TurboQuantMSE(d=128, b=2, seed=7)
        r1 = q1.dequantize(q1.quantize(x))
        r2 = q2.dequantize(q2.quantize(x))
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        x = random_unit_vectors(10, 128, np.random.RandomState(42))
        q1 = TurboQuantMSE(d=128, b=2, seed=0)
        q2 = TurboQuantMSE(d=128, b=2, seed=1)
        r1 = q1.dequantize(q1.quantize(x))
        r2 = q2.dequantize(q2.quantize(x))
        assert not np.allclose(r1, r2)


class TestMSEDistortion:
    """Mean ||x - x_hat||^2 should match paper's expected values."""

    # Paper: D_mse ≈ 0.36, 0.117, 0.03, 0.009 for b=1,2,3,4
    @pytest.mark.parametrize(
        "b, expected_mse",
        [(1, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)],
    )
    def test_mse_matches_paper(self, b, expected_mse):
        d = 256
        n = 1000
        rng = np.random.RandomState(42)
        x = random_unit_vectors(n, d, rng)

        q = TurboQuantMSE(d=d, b=b, seed=0)
        x_hat = q.dequantize(q.quantize(x))

        mse = np.mean(np.sum((x - x_hat) ** 2, axis=1))
        np.testing.assert_allclose(mse, expected_mse, rtol=0.20)


class TestDifferentDimensions:
    """Works across various dimensions."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_round_trip_various_d(self, d):
        rng = np.random.RandomState(42)
        x = random_unit_vectors(50, d, rng)
        q = TurboQuantMSE(d=d, b=2, seed=0)
        x_hat = q.dequantize(q.quantize(x))
        assert x_hat.shape == x.shape
        # MSE should be reasonable (< 0.5 for b=2)
        mse = np.mean(np.sum((x - x_hat) ** 2, axis=1))
        assert mse < 0.5


class TestNonUnitVectors:
    """quantize_with_norm / dequantize_with_norm for non-unit vectors."""

    def test_norm_preserved(self):
        d = 128
        rng = np.random.RandomState(42)
        x = rng.randn(d) * 5.0  # non-unit vector
        original_norm = np.linalg.norm(x)

        q = TurboQuantMSE(d=d, b=3, seed=0)
        quantized, norm = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norm)

        np.testing.assert_allclose(np.linalg.norm(x_hat) / original_norm, 1.0, atol=0.3)

    def test_batch_norm(self):
        d = 128
        rng = np.random.RandomState(42)
        x = rng.randn(10, d) * 3.0

        q = TurboQuantMSE(d=d, b=2, seed=0)
        quantized, norms = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norms)

        assert x_hat.shape == x.shape
        assert norms.shape == (10,)
