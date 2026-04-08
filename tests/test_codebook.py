"""Tests for Lloyd-Max codebook computation."""

import numpy as np
import pytest
from scipy.integrate import quad

from turboquant.codebook import LloydMaxCodebook
from turboquant.utils import beta_pdf, gaussian_approx_pdf


class TestCentroidProperties:
    """Basic structural properties of computed centroids."""

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, b):
        cb = LloydMaxCodebook(d=128, b=b)
        assert len(cb.centroids) == 2**b

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_centroids_sorted(self, b):
        cb = LloydMaxCodebook(d=128, b=b)
        assert np.all(np.diff(cb.centroids) > 0)

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_centroids_within_bounds(self, b):
        cb = LloydMaxCodebook(d=128, b=b)
        assert np.all(cb.centroids >= -1.0)
        assert np.all(cb.centroids <= 1.0)

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_symmetry(self, b):
        """Centroids should be symmetric: c_i ≈ -c_{2^b - 1 - i}."""
        cb = LloydMaxCodebook(d=128, b=b)
        n = 2**b
        for i in range(n // 2):
            np.testing.assert_allclose(
                cb.centroids[i], -cb.centroids[n - 1 - i], atol=1e-10
            )


class TestBoundaryProperties:
    """Properties of Voronoi boundaries."""

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_correct_number_of_boundaries(self, b):
        cb = LloydMaxCodebook(d=128, b=b)
        assert len(cb.boundaries) == 2**b + 1

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_boundary_endpoints(self, b):
        cb = LloydMaxCodebook(d=128, b=b)
        assert cb.boundaries[0] == -1.0
        assert cb.boundaries[-1] == 1.0

    def test_boundaries_sorted(self):
        cb = LloydMaxCodebook(d=128, b=2)
        assert np.all(np.diff(cb.boundaries) > 0)

    def test_boundaries_are_midpoints(self):
        """Interior boundaries should be midpoints of adjacent centroids."""
        cb = LloydMaxCodebook(d=128, b=2)
        for i in range(len(cb.centroids) - 1):
            expected = (cb.centroids[i] + cb.centroids[i + 1]) / 2
            np.testing.assert_allclose(cb.boundaries[i + 1], expected, atol=1e-12)


class TestExpectedCentroidValues:
    """Verify centroids match known values from the paper."""

    def test_b1_centroids_large_d(self):
        """b=1 centroids ≈ ±sqrt(2/(pi*d)) for large d."""
        d = 512
        cb = LloydMaxCodebook(d=d, b=1)
        expected = np.sqrt(2.0 / (np.pi * d))
        np.testing.assert_allclose(cb.centroids[0], -expected, rtol=0.05)
        np.testing.assert_allclose(cb.centroids[1], expected, rtol=0.05)

    def test_b2_centroids_large_d(self):
        """b=2 centroids ≈ {±0.453, ±1.510} / sqrt(d)."""
        d = 512
        cb = LloydMaxCodebook(d=d, b=2)
        sd = np.sqrt(d)
        scaled = cb.centroids * sd
        np.testing.assert_allclose(np.abs(scaled[0]), 1.510, rtol=0.05)
        np.testing.assert_allclose(np.abs(scaled[1]), 0.453, rtol=0.05)


class TestQuantizeDequantize:
    """Round-trip quantization tests."""

    def test_round_trip_returns_centroids(self):
        """dequantize(quantize(v)) should return nearest centroid."""
        cb = LloydMaxCodebook(d=128, b=2)
        values = np.array([-0.05, 0.0, 0.05, 0.1])
        indices = cb.quantize(values)
        reconstructed = cb.dequantize(indices)
        # Each reconstructed value must be one of the centroids
        for v in reconstructed:
            assert v in cb.centroids

    def test_quantize_indices_range(self):
        cb = LloydMaxCodebook(d=128, b=3)
        values = np.linspace(-0.2, 0.2, 100)
        indices = cb.quantize(values)
        assert np.all(indices >= 0)
        assert np.all(indices < 2**3)

    def test_quantize_maps_to_nearest(self):
        """Values equal to a centroid should map to that centroid's index."""
        cb = LloydMaxCodebook(d=128, b=2)
        indices = cb.quantize(cb.centroids)
        reconstructed = cb.dequantize(indices)
        np.testing.assert_allclose(reconstructed, cb.centroids, atol=1e-12)

    def test_quantize_batch_shape(self):
        """Quantize should handle 2D input (n, d) element-wise."""
        cb = LloydMaxCodebook(d=128, b=2)
        values = np.random.RandomState(42).randn(10, 128) / np.sqrt(128)
        indices = cb.quantize(values)
        assert indices.shape == values.shape
        reconstructed = cb.dequantize(indices)
        assert reconstructed.shape == values.shape


class TestMSECost:
    """Verify MSE distortion cost d * C(f_X, b) matches paper values."""

    # Paper Table: d * C(f_X, b) ≈ 0.36, 0.117, 0.03, 0.009 for b=1,2,3,4
    @pytest.mark.parametrize(
        "b, expected_d_times_cost",
        [(1, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)],
    )
    def test_mse_cost_matches_paper(self, b, expected_d_times_cost):
        d = 512
        cb = LloydMaxCodebook(d=d, b=b)
        d_times_cost = d * cb.mse_cost
        np.testing.assert_allclose(d_times_cost, expected_d_times_cost, rtol=0.20)


class TestDifferentDimensions:
    """Codebook should work across dimensions."""

    @pytest.mark.parametrize("d", [32, 64, 128, 256, 512])
    def test_converges_for_various_d(self, d):
        cb = LloydMaxCodebook(d=d, b=2)
        assert len(cb.centroids) == 4
        assert np.all(np.diff(cb.centroids) > 0)

    def test_use_gaussian_approx_for_large_d(self):
        """For d >= 50, gaussian approx should yield similar centroids."""
        cb_exact = LloydMaxCodebook(d=128, b=2, use_gaussian_approx=False)
        cb_approx = LloydMaxCodebook(d=128, b=2, use_gaussian_approx=True)
        np.testing.assert_allclose(cb_exact.centroids, cb_approx.centroids, rtol=0.02)
