import numpy as np
import pytest

from src.conv import convolve


@pytest.mark.repeat(10)  # repeat the test 10 times
@pytest.mark.parametrize("mode", ["full", "valid"])
def test_conv(mode):
    x = np.random.randn(10)
    k = np.random.randn(3)

    assert np.allclose(np.convolve(x, k, mode=mode), convolve(x, k, mode=mode))  # type: ignore
