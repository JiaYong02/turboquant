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
