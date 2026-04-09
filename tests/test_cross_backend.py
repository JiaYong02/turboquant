"""Cross-backend statistical equivalence tests.

Verifies that NumPy and PyTorch backends produce statistically similar
distortion characteristics (not identical values, since RNGs differ).
"""

import numpy as np
import pytest
import torch

from turboquant.quantizer_mse import TurboQuantMSE
from turboquant.quantizer_mse_torch import TurboQuantMSETorch
from turboquant.quantizer_prod import TurboQuantProd
from turboquant.quantizer_prod_torch import TurboQuantProdTorch
from turboquant.utils import random_unit_vectors
from conftest import random_unit_vectors_torch


class TestMSECrossBackend:
    """Both backends should achieve similar MSE distortion for same (d, b)."""

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_mse_distributions_similar(self, b):
        d = 256
        n = 500

        # NumPy
        rng = np.random.RandomState(42)
        x_np = random_unit_vectors(n, d, rng)
        q_np = TurboQuantMSE(d=d, b=b, seed=0)
        x_hat_np = q_np.dequantize(q_np.quantize(x_np))
        mse_np = np.mean(np.sum((x_np - x_hat_np) ** 2, axis=1))

        # PyTorch
        x_torch = random_unit_vectors_torch(n, d, seed=42)
        q_torch = TurboQuantMSETorch(d=d, b=b, seed=0)
        x_hat_torch = q_torch.dequantize(q_torch.quantize(x_torch))
        mse_torch = torch.mean(torch.sum((x_torch - x_hat_torch) ** 2, dim=1)).item()

        # Both should be close to the paper's expected value
        # Allow 30% relative difference between backends
        ratio = mse_torch / mse_np if mse_np > 0 else 1.0
        assert 0.7 < ratio < 1.3, (
            f"b={b}: NumPy MSE={mse_np:.4f}, PyTorch MSE={mse_torch:.4f}, "
            f"ratio={ratio:.3f}"
        )


class TestProdCrossBackend:
    """Both backends should achieve similar inner-product distortion."""

    @pytest.mark.parametrize("b", [2, 3])
    def test_prod_unbiasedness_both_backends(self, b):
        d = 128
        n_seeds = 200

        # NumPy
        rng = np.random.RandomState(42)
        x_np = random_unit_vectors(1, d, rng)[0]
        y_np = random_unit_vectors(1, d, rng)[0]
        true_ip_np = float(x_np @ y_np)

        np_estimates = np.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProd(d=d, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x_np))
            np_estimates[seed] = y_np @ x_hat

        # PyTorch
        x_t = random_unit_vectors_torch(1, d, seed=42)[0]
        y_t = random_unit_vectors_torch(1, d, seed=43)[0]
        true_ip_t = (x_t @ y_t).item()

        torch_estimates = torch.empty(n_seeds)
        for seed in range(n_seeds):
            q = TurboQuantProdTorch(d=d, b=b, seed=seed)
            x_hat = q.dequantize(q.quantize(x_t))
            torch_estimates[seed] = y_t @ x_hat

        # Both should be unbiased (mean ≈ true IP)
        np_mean = float(np.mean(np_estimates))
        torch_mean = float(torch_estimates.mean())

        assert abs(np_mean - true_ip_np) < 0.05
        assert abs(torch_mean - true_ip_t) < 0.05
