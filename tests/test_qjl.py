"""Tests for QJL (Quantized Johnson-Lindenstrauss) 1-bit quantizer."""

import numpy as np
import pytest

from turboquant.qjl import QJL
from turboquant.utils import random_unit_vectors


class TestQuantizeOutput:
    """Output format and value constraints."""

    def test_output_values_in_minus1_plus1(self):
        """All quantized values must be in {-1, +1}, never 0."""
        qjl = QJL(d=128, seed=0)
        rng = np.random.RandomState(42)
        x = rng.randn(128)
        signs = qjl.quantize(x)
        unique = set(np.unique(signs))
        assert unique.issubset({-1, 1})
        assert 0 not in unique

    def test_output_dtype_int8(self):
        qjl = QJL(d=64, seed=0)
        signs = qjl.quantize(np.random.RandomState(0).randn(64))
        assert signs.dtype == np.int8

    def test_zero_input_maps_to_plus1(self):
        """sign(0) should map to +1, not 0."""
        qjl = QJL(d=64, seed=0)
        signs = qjl.quantize(np.zeros(64))
        assert np.all(signs == 1)


class TestShapes:
    """Shape preservation for single and batch inputs."""

    def test_single_vector_quantize(self):
        qjl = QJL(d=128, seed=0)
        x = np.random.randn(128)
        signs = qjl.quantize(x)
        assert signs.shape == (128,)

    def test_batch_quantize(self):
        qjl = QJL(d=128, seed=0)
        x = np.random.randn(10, 128)
        signs = qjl.quantize(x)
        assert signs.shape == (10, 128)

    def test_single_vector_dequantize(self):
        qjl = QJL(d=128, seed=0)
        signs = np.ones(128, dtype=np.int8)
        result = qjl.dequantize(signs, gamma=1.0)
        assert result.shape == (128,)

    def test_batch_dequantize(self):
        qjl = QJL(d=128, seed=0)
        signs = np.ones((10, 128), dtype=np.int8)
        gamma = np.ones(10)
        result = qjl.dequantize(signs, gamma)
        assert result.shape == (10, 128)


class TestDeterminism:
    """Same seed produces same results."""

    def test_same_seed_same_matrix(self):
        qjl1 = QJL(d=64, seed=42)
        qjl2 = QJL(d=64, seed=42)
        np.testing.assert_array_equal(qjl1.S, qjl2.S)

    def test_different_seeds_differ(self):
        qjl1 = QJL(d=64, seed=0)
        qjl2 = QJL(d=64, seed=1)
        assert not np.allclose(qjl1.S, qjl2.S)

    def test_quantize_deterministic(self):
        qjl = QJL(d=64, seed=42)
        x = np.random.randn(64)
        s1 = qjl.quantize(x)
        s2 = qjl.quantize(x)
        np.testing.assert_array_equal(s1, s2)


class TestUnbiasedness:
    """E[<y, dequant(quant(x), ||x||)>] ≈ <y, x> — the key QJL property."""

    def test_inner_product_unbiased(self):
        """Statistical test: average inner product should match true value."""
        d = 128
        n_trials = 5000
        rng = np.random.RandomState(42)

        # Fixed x and y
        x = rng.randn(d)
        x = x / np.linalg.norm(x)
        y = rng.randn(d)
        y = y / np.linalg.norm(y)

        true_ip = x @ y
        gamma = np.linalg.norm(x)  # = 1.0 since x is unit

        estimates = np.empty(n_trials)
        for i in range(n_trials):
            qjl = QJL(d=d, seed=i)
            signs = qjl.quantize(x)
            x_hat = qjl.dequantize(signs, gamma)
            estimates[i] = y @ x_hat

        mean_estimate = np.mean(estimates)
        # Should be close to true inner product
        np.testing.assert_allclose(mean_estimate, true_ip, atol=0.03)

    def test_unbiased_non_unit_vector(self):
        """Unbiasedness should hold for non-unit vectors too."""
        d = 128
        n_trials = 5000
        rng = np.random.RandomState(99)

        x = rng.randn(d) * 3.0  # non-unit
        y = rng.randn(d)
        true_ip = x @ y
        gamma = np.linalg.norm(x)

        estimates = np.empty(n_trials)
        for i in range(n_trials):
            qjl = QJL(d=d, seed=i)
            signs = qjl.quantize(x)
            x_hat = qjl.dequantize(signs, gamma)
            estimates[i] = y @ x_hat

        mean_estimate = np.mean(estimates)
        np.testing.assert_allclose(mean_estimate, true_ip, rtol=0.08)


class TestVarianceBound:
    """Var(<y, reconstructed>) <= pi/(2d) * ||x||^2 * ||y||^2."""

    def test_variance_within_bound(self):
        d = 128
        n_trials = 5000
        rng = np.random.RandomState(42)

        x = rng.randn(d)
        y = rng.randn(d)
        gamma = np.linalg.norm(x)

        estimates = np.empty(n_trials)
        for i in range(n_trials):
            qjl = QJL(d=d, seed=i)
            signs = qjl.quantize(x)
            x_hat = qjl.dequantize(signs, gamma)
            estimates[i] = y @ x_hat

        empirical_var = np.var(estimates)
        theoretical_bound = (np.pi / (2 * d)) * np.linalg.norm(x) ** 2 * np.linalg.norm(y) ** 2

        # Allow 30% margin for statistical noise
        assert empirical_var <= theoretical_bound * 1.3, (
            f"Variance {empirical_var:.6f} exceeds bound {theoretical_bound:.6f} * 1.3"
        )
