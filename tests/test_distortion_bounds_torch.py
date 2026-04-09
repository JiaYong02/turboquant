"""Distortion bound validation for PyTorch backend.

Mirrors test_distortion_bounds.py — verifies that empirical MSE and
inner-product distortions fall within the paper's proven bounds.
"""

import numpy as np
import pytest
import torch

from turboquant.quantizer_mse_torch import TurboQuantMSETorch
from turboquant.quantizer_prod_torch import TurboQuantProdTorch
from conftest import random_unit_vectors_torch

D = 512
N_VECTORS = 1000


class TestMSEDistortionBounds:
    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_mse_within_bounds(self, b):
        x = random_unit_vectors_torch(N_VECTORS, D, seed=42)

        q = TurboQuantMSETorch(d=D, b=b, seed=0)
        x_hat = q.dequantize(q.quantize(x))

        d_mse = torch.mean(torch.sum((x - x_hat) ** 2, dim=1)).item()

        lower = 1.0 / (4**b)
        upper = np.sqrt(3) * np.pi / 2 * (1.0 / (4**b))

        assert d_mse >= lower * 0.8, (
            f"b={b}: D_mse={d_mse:.6f} below lower bound {lower:.6f}"
        )
        assert d_mse <= upper * 1.2, (
            f"b={b}: D_mse={d_mse:.6f} above upper bound {upper:.6f}"
        )


class TestProdDistortionBounds:
    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_prod_within_bounds(self, b):
        n_seeds = 300
        x = random_unit_vectors_torch(1, D, seed=42)[0]
        y = random_unit_vectors_torch(1, D, seed=43)[0]
        true_ip = (x @ y).item()

        errors_sq = torch.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProdTorch(d=D, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            errors_sq[seed] = (y @ x_hat - true_ip) ** 2

        d_prod = errors_sq.mean().item()

        lower = 1.0 / (D * 4**b)
        upper = np.sqrt(3) * np.pi**2 / (2 * D) * (1.0 / 4**b)

        assert d_prod >= lower * 0.2, (
            f"b={b}: D_prod={d_prod:.8f} below lower bound {lower:.8f}"
        )
        assert d_prod <= upper * 2.0, (
            f"b={b}: D_prod={d_prod:.8f} above upper bound {upper:.8f}"
        )


class TestProdUnbiasedness:
    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_mean_error_centered_at_zero(self, b):
        n_seeds = 300
        x = random_unit_vectors_torch(1, D, seed=42)[0]
        y = random_unit_vectors_torch(1, D, seed=43)[0]
        true_ip = (x @ y).item()

        errors = torch.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProdTorch(d=D, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x))
            errors[seed] = y @ x_hat - true_ip

        mean_error = errors.mean().item()
        assert abs(mean_error) < 0.02, f"b={b}: mean_error={mean_error:.4f}"


class TestDistortionDecreasing:
    def test_mse_distortion_decreases_with_bits(self):
        x = random_unit_vectors_torch(500, D, seed=42)

        distortions = []
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSETorch(d=D, b=b, seed=0)
            x_hat = q.dequantize(q.quantize(x))
            mse = torch.mean(torch.sum((x - x_hat) ** 2, dim=1)).item()
            distortions.append(mse)

        for i in range(len(distortions) - 1):
            assert distortions[i + 1] < distortions[i], (
                f"D_mse did not decrease: b={i+1}={distortions[i]:.4f}, "
                f"b={i+2}={distortions[i+1]:.4f}"
            )

    def test_prod_distortion_decreases_with_bits(self):
        distortions = []
        for b in [1, 2, 3]:
            q = TurboQuantProdTorch(d=D, b=b, seed=0)
            x = random_unit_vectors_torch(500, D, seed=42)
            y = random_unit_vectors_torch(500, D, seed=99)

            x_hat = q.dequantize(q.quantize(x))
            estimated_ips = torch.sum(y * x_hat, dim=1)
            true_ips = torch.sum(y * x, dim=1)
            distortions.append(torch.mean((estimated_ips - true_ips) ** 2).item())

        for i in range(len(distortions) - 1):
            assert distortions[i + 1] < distortions[i]
