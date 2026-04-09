"""Tests for LloydMaxCodebookTorch."""

import numpy as np
import pytest
import torch

from turboquant.codebook import LloydMaxCodebook
from turboquant.codebook_torch import LloydMaxCodebookTorch


class TestCentroidsMatchNumpy:
    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_centroids_match(self, b):
        d = 128
        np_cb = LloydMaxCodebook(d, b)
        torch_cb = LloydMaxCodebookTorch(d, b)
        np.testing.assert_allclose(
            torch_cb.centroids.numpy(), np_cb.centroids, atol=1e-6
        )

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_boundaries_match(self, b):
        d = 128
        np_cb = LloydMaxCodebook(d, b)
        torch_cb = LloydMaxCodebookTorch(d, b)
        np.testing.assert_allclose(
            torch_cb.boundaries.numpy(), np_cb.boundaries, atol=1e-6
        )

    def test_mse_cost_matches(self):
        np_cb = LloydMaxCodebook(128, 2)
        torch_cb = LloydMaxCodebookTorch(128, 2)
        assert abs(torch_cb.mse_cost - np_cb.mse_cost) < 1e-10


class TestQuantizeDequantize:
    def test_round_trip_produces_centroids(self):
        cb = LloydMaxCodebookTorch(128, 2)
        values = torch.linspace(-0.9, 0.9, 100)
        indices = cb.quantize(values)
        reconstructed = cb.dequantize(indices)
        # Every reconstructed value must be one of the centroids
        for v in reconstructed:
            assert v.item() in cb.centroids.tolist()

    def test_output_dtype_uint8(self):
        cb = LloydMaxCodebookTorch(128, 4)
        values = torch.randn(50)
        indices = cb.quantize(values)
        assert indices.dtype == torch.uint8


class TestSearchsortedBoundary:
    def test_extreme_values(self):
        cb = LloydMaxCodebookTorch(128, 2)
        # Very negative → index 0, very positive → last index
        indices = cb.quantize(torch.tensor([-10.0, 10.0]))
        assert indices[0].item() == 0
        assert indices[1].item() == cb._n_centroids - 1

    def test_at_boundary(self):
        cb = LloydMaxCodebookTorch(128, 2)
        # Value exactly at a boundary midpoint
        mid = (cb.centroids[0] + cb.centroids[1]) / 2
        idx = cb.quantize(mid.unsqueeze(0))
        assert idx[0].item() in [0, 1]


class TestDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_placement(self):
        cb = LloydMaxCodebookTorch(128, 2, device="cuda")
        assert cb.centroids.device.type == "cuda"
        values = torch.randn(10, device="cuda")
        indices = cb.quantize(values)
        assert indices.device.type == "cuda"
        recon = cb.dequantize(indices)
        assert recon.device.type == "cuda"
