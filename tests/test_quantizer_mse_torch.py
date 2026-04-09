"""Tests for TurboQuantMSETorch (Algorithm 1)."""

import pytest
import torch

from turboquant.quantizer_mse_torch import TurboQuantMSETorch
from conftest import random_unit_vectors_torch


class TestShapes:
    def test_single_vector(self):
        q = TurboQuantMSETorch(d=128, b=2, seed=0)
        x = random_unit_vectors_torch(1, 128, seed=0)[0]
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (128,)

    def test_batch(self):
        q = TurboQuantMSETorch(d=128, b=2, seed=0)
        x = random_unit_vectors_torch(20, 128, seed=0)
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.shape == (20, 128)

    def test_idx_dtype_uint8(self):
        q = TurboQuantMSETorch(d=64, b=4, seed=0)
        x = random_unit_vectors_torch(5, 64, seed=0)
        quantized = q.quantize(x)
        assert quantized["idx"].dtype == torch.uint8


class TestDeterminism:
    def test_same_seed_same_output(self):
        x = random_unit_vectors_torch(10, 128, seed=42)
        q1 = TurboQuantMSETorch(d=128, b=2, seed=7)
        q2 = TurboQuantMSETorch(d=128, b=2, seed=7)
        r1 = q1.dequantize(q1.quantize(x))
        r2 = q2.dequantize(q2.quantize(x))
        torch.testing.assert_close(r1, r2)

    def test_different_seeds_differ(self):
        x = random_unit_vectors_torch(10, 128, seed=42)
        q1 = TurboQuantMSETorch(d=128, b=2, seed=0)
        q2 = TurboQuantMSETorch(d=128, b=2, seed=1)
        r1 = q1.dequantize(q1.quantize(x))
        r2 = q2.dequantize(q2.quantize(x))
        assert not torch.allclose(r1, r2)


class TestMSEDistortion:
    @pytest.mark.parametrize(
        "b, expected_mse",
        [(1, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)],
    )
    def test_mse_matches_paper(self, b, expected_mse):
        d = 256
        n = 1000
        x = random_unit_vectors_torch(n, d, seed=42)

        q = TurboQuantMSETorch(d=d, b=b, seed=0)
        x_hat = q.dequantize(q.quantize(x))

        mse = torch.mean(torch.sum((x - x_hat) ** 2, dim=1)).item()
        assert abs(mse - expected_mse) / expected_mse < 0.25, (
            f"b={b}: MSE={mse:.4f}, expected≈{expected_mse}"
        )


class TestDifferentDimensions:
    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_round_trip_various_d(self, d):
        x = random_unit_vectors_torch(50, d, seed=42)
        q = TurboQuantMSETorch(d=d, b=2, seed=0)
        x_hat = q.dequantize(q.quantize(x))
        assert x_hat.shape == x.shape
        mse = torch.mean(torch.sum((x - x_hat) ** 2, dim=1)).item()
        assert mse < 0.5


class TestNonUnitVectors:
    def test_norm_preserved(self):
        d = 128
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(d, generator=gen) * 5.0
        original_norm = torch.linalg.norm(x).item()

        q = TurboQuantMSETorch(d=d, b=3, seed=0)
        quantized, norm = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norm)

        ratio = torch.linalg.norm(x_hat).item() / original_norm
        assert abs(ratio - 1.0) < 0.3

    def test_batch_norm(self):
        d = 128
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(10, d, generator=gen) * 3.0

        q = TurboQuantMSETorch(d=d, b=2, seed=0)
        quantized, norms = q.quantize_with_norm(x)
        x_hat = q.dequantize_with_norm(quantized, norms)

        assert x_hat.shape == x.shape
        assert norms.shape == (10,)


class TestDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_round_trip(self):
        d = 128
        q = TurboQuantMSETorch(d=d, b=2, seed=0, device="cuda")
        x = random_unit_vectors_torch(10, d, seed=42, device="cuda")
        quantized = q.quantize(x)
        x_hat = q.dequantize(quantized)
        assert x_hat.device.type == "cuda"
        assert x_hat.shape == (10, d)
