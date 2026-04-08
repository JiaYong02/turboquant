"""Tests for RandomRotation."""

import numpy as np
import pytest
from scipy import stats

from turboquant.rotation import RandomRotation
from turboquant.utils import beta_pdf, random_unit_vectors


class TestRandomRotation:
    def test_orthogonality(self, dimension):
        rot = RandomRotation(dimension, seed=0)
        identity = rot.Pi @ rot.Pi.T
        np.testing.assert_allclose(identity, np.eye(dimension), atol=1e-12)

    def test_norm_preservation(self, dimension, rng):
        rot = RandomRotation(dimension, seed=0)
        x = rng.randn(dimension)
        y = rot.rotate(x)
        np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x), rtol=1e-12)

    def test_round_trip_single(self, dimension, rng):
        rot = RandomRotation(dimension, seed=0)
        x = rng.randn(dimension)
        reconstructed = rot.unrotate(rot.rotate(x))
        np.testing.assert_allclose(reconstructed, x, atol=1e-12)

    def test_round_trip_batch(self, dimension, rng):
        rot = RandomRotation(dimension, seed=0)
        x = rng.randn(10, dimension)
        reconstructed = rot.unrotate(rot.rotate(x))
        np.testing.assert_allclose(reconstructed, x, atol=1e-12)

    def test_determinism(self, dimension):
        rot1 = RandomRotation(dimension, seed=99)
        rot2 = RandomRotation(dimension, seed=99)
        np.testing.assert_array_equal(rot1.Pi, rot2.Pi)

    def test_different_seeds_differ(self, dimension):
        rot1 = RandomRotation(dimension, seed=0)
        rot2 = RandomRotation(dimension, seed=1)
        assert not np.allclose(rot1.Pi, rot2.Pi)

    def test_coordinate_distribution(self):
        """Rotated unit vector coordinates should follow Beta distribution (Lemma 1)."""
        d = 128
        n_samples = 5000
        rot = RandomRotation(d, seed=0)
        vectors = random_unit_vectors(n_samples, d, rng=np.random.RandomState(1))
        rotated = rot.rotate(vectors)  # (n_samples, d)

        # Collect first coordinate across all samples
        coords = rotated[:, 0]

        # Build the CDF from our beta_pdf via numerical integration
        from scipy.integrate import quad

        def cdf_scalar(t):
            val, _ = quad(lambda x: beta_pdf(x, d), -1.0, t)
            return val

        def cdf(t):
            t = np.atleast_1d(t)
            return np.array([cdf_scalar(float(v)) for v in t])

        # K-S test: compare empirical distribution to theoretical CDF
        ks_stat, p_value = stats.kstest(coords, cdf)
        # With 5000 samples, p-value should be well above 0.01
        assert p_value > 0.01, f"K-S test failed: stat={ks_stat:.4f}, p={p_value:.4f}"

    def test_output_shape_single(self):
        rot = RandomRotation(64, seed=0)
        x = np.ones(64)
        assert rot.rotate(x).shape == (64,)
        assert rot.unrotate(x).shape == (64,)

    def test_output_shape_batch(self):
        rot = RandomRotation(64, seed=0)
        x = np.ones((5, 64))
        assert rot.rotate(x).shape == (5, 64)
        assert rot.unrotate(x).shape == (5, 64)
