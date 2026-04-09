"""Shared test fixtures for TurboQuant tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture(params=[64, 128, 256])
def dimension(request):
    """Test across multiple dimensions."""
    return request.param


try:
    import torch

    @pytest.fixture
    def torch_device():
        """Best available torch device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def random_unit_vectors_torch(
        n: int, d: int, seed: int = 42, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Generate n random unit vectors as a torch tensor."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(n, d, generator=gen, dtype=torch.float32)
        norms = torch.linalg.norm(x, dim=1, keepdim=True)
        x = x / norms
        if device is not None:
            x = x.to(device)
        return x

except ImportError:
    pass
