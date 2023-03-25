import numpy as np


def sample_hold_kernel(x: np.array, offset: float, width: float) -> np.array:
    """Sample and hold interpolation kernel"""
    x = x - offset
    return (x >= 0) * (x < width)


def nearest_neighbour_kernel(x: np.array, offset: float, width: float) -> np.array:
    """Nearest neighbour interpolation kernel"""
    x = x - offset
    return (x >= (-1 * width / 2)) * (x < width / 2)


def linear_kernel(x: np.array, offset: float, width: float) -> np.array:
    """Linear interpolation kernel"""
    x = x - offset
    x = x / width
    return (1 - np.abs(x)) * (np.abs(x) < 1)


def sinc_kernel(x: np.array, offset: float, width: float, alpha: float = np.inf) -> np.array:
    """Normalized sine interpolation kernel"""
    x = x - offset
    x = x / width
    return (x >= -alpha) * (x < alpha) * np.sinc(x)


def keys_kernel(x: np.array, offset: float, width: float, alpha: float = -0.5) -> np.array:
    """
    Interpolation kernel given by Keys bi-cubic function

    :references:
        * https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        * http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
    """
    x = x - offset
    x = x / width
    x = np.abs(x)
    return ((alpha + 2) * x**3 - (alpha + 3) * x**2 + 1) * (x >= 0) * (x < 1) + (
        alpha * x**3 - 5 * alpha * x**2 + 8 * alpha * x - 4 * alpha
    ) * (x >= 1) * (x < 2) * (1 - (x >= 0) * (x < 1))
