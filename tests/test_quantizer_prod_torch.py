"""Tests for TurboQuantProdTorch (Algorithm 2)."""

import pytest
import torch

from turboquant.quantizer_prod_torch import TurboQuantProdTorch
from conftest import random_unit_vectors_torch


class TestShapes:
    def test_single_vector(self):
        q = TurboQuantProdTorch(d=128, b=2, seed=0)
        x = random_unit_vectors_torch(1, 128, seed=0)[0]
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (128,)

    def test_batch(self):
        q = TurboQuantProdTorch(d=128, b=2, seed=0)
        x = random_unit_vectors_torch(20, 128, seed=0)
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (20, 128)


class TestB1SpecialCase:
    def test_no_mse_stage(self):
        q = TurboQuantProdTorch(d=128, b=1, seed=0)
        assert q.mse is None

    def test_mse_field_is_none(self):
        q = TurboQuantProdTorch(d=128, b=1, seed=0)
        x = random_unit_vectors_torch(5, 128, seed=0)
        quantized = q.quantize(x)
        assert quantized["mse"] is None

    def test_b1_still_works(self):
        q = TurboQuantProdTorch(d=128, b=1, seed=0)
        x = random_unit_vectors_torch(10, 128, seed=0)
        x_hat = q.dequantize(q.quantize(x))
        assert x_hat.shape == x.shape


class TestUnbiasedness:
    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_inner_product_unbiased(self, b):
        d = 128
        n_seeds = 200
        x = random_unit_vectors_torch(1, d, seed=42)[0]
        y = random_unit_vectors_torch(1, d, seed=43)[0]
        true_ip = (x @ y).item()

        estimates = torch.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProdTorch(d=d, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            estimates[seed] = y @ x_hat

        mean_estimate = estimates.mean().item()
        assert abs(mean_estimate - true_ip) < 0.05, (
            f"b={b}: mean={mean_estimate:.4f}, true={true_ip:.4f}"
        )


class TestInnerProductDistortion:
    @pytest.mark.parametrize(
        "b, expected_d_times_dprod",
        [(1, 1.57), (2, 0.56), (3, 0.18)],
    )
    def test_distortion_matches_paper(self, b, expected_d_times_dprod):
        d = 256
        n_seeds = 500
        x = random_unit_vectors_torch(1, d, seed=42)[0]
        y = random_unit_vectors_torch(1, d, seed=43)[0]
        true_ip = (x @ y).item()

        errors_sq = torch.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProdTorch(d=d, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            errors_sq[seed] = (y @ x_hat - true_ip) ** 2

        empirical = d * errors_sq.mean().item()
        assert abs(empirical - expected_d_times_dprod) / expected_d_times_dprod < 0.45, (
            f"b={b}: d*D_prod={empirical:.4f}, expected≈{expected_d_times_dprod}"
        )


class TestDifferentDimensions:
    @pytest.mark.parametrize("d", [128, 256])
    def test_round_trip_various_d(self, d):
        x = random_unit_vectors_torch(20, d, seed=42)
        q = TurboQuantProdTorch(d=d, b=2, seed=0)
        x_hat = q.dequantize(q.quantize(x))
        assert x_hat.shape == x.shape


class TestQuantizeWithNorm:
    def test_single_vector(self):
        q = TurboQuantProdTorch(d=128, b=2, seed=0)
        x = torch.randn(128) * 3.0
        quantized, norm = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norm)
        assert x_hat.shape == (128,)
        assert norm.item() == pytest.approx(torch.linalg.norm(x).item(), rel=1e-5)

    def test_batch(self):
        q = TurboQuantProdTorch(d=128, b=2, seed=0)
        x = torch.randn(20, 128) * 5.0
        quantized, norms = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norms)
        assert x_hat.shape == (20, 128)
        assert norms.shape == (20,)

    def test_zero_vector(self):
        q = TurboQuantProdTorch(d=128, b=2, seed=0)
        x = torch.zeros(128)
        quantized, norm = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norm)
        assert norm.item() == 0.0
        assert x_hat.shape == (128,)

    def test_norm_preserved(self):
        q = TurboQuantProdTorch(d=128, b=4, seed=42)
        x = torch.randn(50, 128) * 2.0
        quantized, norms = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norms)
        # Reconstructed norms should be close to original
        hat_norms = torch.linalg.norm(x_hat, dim=1)
        ratio = hat_norms / norms
        assert (ratio.mean() - 1.0).abs() < 0.3


class TestDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_round_trip(self):
        d = 128
        q = TurboQuantProdTorch(d=d, b=2, seed=0, device="cuda")
        x = random_unit_vectors_torch(10, d, seed=42, device="cuda")
        x_hat = q.dequantize(q.quantize(x))
        assert x_hat.device.type == "cuda"
