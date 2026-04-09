"""Tests for RandomRotationTorch."""

import pytest
import torch

from turboquant.rotation_torch import RandomRotationTorch
from conftest import random_unit_vectors_torch


class TestOrthogonality:
    def test_orthogonal(self, dimension):
        rot = RandomRotationTorch(dimension, seed=0)
        identity = rot.Pi @ rot.Pi.T
        torch.testing.assert_close(
            identity, torch.eye(dimension), atol=1e-5, rtol=1e-5
        )


class TestNormPreservation:
    def test_norm_preserved(self, dimension):
        rot = RandomRotationTorch(dimension, seed=0)
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(dimension, generator=gen)
        y = rot.rotate(x)
        torch.testing.assert_close(
            torch.linalg.norm(y), torch.linalg.norm(x), atol=1e-5, rtol=1e-5
        )


class TestRoundTrip:
    def test_single(self, dimension):
        rot = RandomRotationTorch(dimension, seed=0)
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(dimension, generator=gen)
        reconstructed = rot.unrotate(rot.rotate(x))
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_batch(self, dimension):
        rot = RandomRotationTorch(dimension, seed=0)
        gen = torch.Generator().manual_seed(42)
        x = torch.randn(10, dimension, generator=gen)
        reconstructed = rot.unrotate(rot.rotate(x))
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)


class TestDeterminism:
    def test_same_seed(self, dimension):
        rot1 = RandomRotationTorch(dimension, seed=99)
        rot2 = RandomRotationTorch(dimension, seed=99)
        torch.testing.assert_close(rot1.Pi, rot2.Pi)

    def test_different_seeds_differ(self, dimension):
        rot1 = RandomRotationTorch(dimension, seed=0)
        rot2 = RandomRotationTorch(dimension, seed=1)
        assert not torch.allclose(rot1.Pi, rot2.Pi)


class TestShapes:
    def test_single(self):
        rot = RandomRotationTorch(64, seed=0)
        x = torch.ones(64)
        assert rot.rotate(x).shape == (64,)
        assert rot.unrotate(x).shape == (64,)

    def test_batch(self):
        rot = RandomRotationTorch(64, seed=0)
        x = torch.ones(5, 64)
        assert rot.rotate(x).shape == (5, 64)
        assert rot.unrotate(x).shape == (5, 64)


class TestDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_placement(self):
        rot = RandomRotationTorch(64, seed=0, device="cuda")
        assert rot.Pi.device.type == "cuda"
        x = torch.randn(64, device="cuda")
        y = rot.rotate(x)
        assert y.device.type == "cuda"
