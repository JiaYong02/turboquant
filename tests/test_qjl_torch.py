"""Tests for QJLTorch."""

import pytest
import torch

from turboquant.qjl_torch import QJLTorch


class TestQuantizeOutput:
    def test_values_in_minus1_plus1(self):
        qjl = QJLTorch(d=128, seed=0)
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(128, generator=gen)
        signs = qjl.quantize(x)
        unique = set(signs.unique().tolist())
        assert unique.issubset({-1, 1})
        assert 0 not in unique

    def test_dtype_int8(self):
        qjl = QJLTorch(d=64, seed=0)
        gen = torch.Generator().manual_seed(0)
        signs = qjl.quantize(torch.randn(64, generator=gen))
        assert signs.dtype == torch.int8

    def test_zero_input_maps_to_plus1(self):
        qjl = QJLTorch(d=64, seed=0)
        signs = qjl.quantize(torch.zeros(64))
        assert torch.all(signs == 1)


class TestShapes:
    def test_single_quantize(self):
        qjl = QJLTorch(d=128, seed=0)
        x = torch.randn(128)
        assert qjl.quantize(x).shape == (128,)

    def test_batch_quantize(self):
        qjl = QJLTorch(d=128, seed=0)
        x = torch.randn(10, 128)
        assert qjl.quantize(x).shape == (10, 128)

    def test_single_dequantize(self):
        qjl = QJLTorch(d=128, seed=0)
        signs = torch.ones(128, dtype=torch.int8)
        result = qjl.dequantize(signs, gamma=1.0)
        assert result.shape == (128,)

    def test_batch_dequantize(self):
        qjl = QJLTorch(d=128, seed=0)
        signs = torch.ones(10, 128, dtype=torch.int8)
        gamma = torch.ones(10)
        result = qjl.dequantize(signs, gamma)
        assert result.shape == (10, 128)


class TestDeterminism:
    def test_same_seed_same_matrix(self):
        qjl1 = QJLTorch(d=64, seed=42)
        qjl2 = QJLTorch(d=64, seed=42)
        torch.testing.assert_close(qjl1.S, qjl2.S)

    def test_different_seeds_differ(self):
        qjl1 = QJLTorch(d=64, seed=0)
        qjl2 = QJLTorch(d=64, seed=1)
        assert not torch.allclose(qjl1.S, qjl2.S)

    def test_quantize_deterministic(self):
        qjl = QJLTorch(d=64, seed=42)
        x = torch.randn(64)
        s1 = qjl.quantize(x)
        s2 = qjl.quantize(x)
        torch.testing.assert_close(s1, s2)


class TestDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_placement(self):
        qjl = QJLTorch(d=64, seed=0, device="cuda")
        assert qjl.S.device.type == "cuda"
        x = torch.randn(64, device="cuda")
        signs = qjl.quantize(x)
        assert signs.device.type == "cuda"
        recon = qjl.dequantize(signs, gamma=1.0)
        assert recon.device.type == "cuda"
