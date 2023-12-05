import numpy as np
from numpy.typing import NDArray
from scipy import stats


def blur_kernel(size: int) -> NDArray:
    """
    Creates a blur kernel of a given size
    When used as convolution kernel it will compute local mean for each group of pixels
    """
    return np.ones((size, size)) / (size**2)


def gaussian_blur_kernel(size: int, sigma: float = 1.0) -> NDArray:
    """
    Creates a gaussian blur kernel of a given size
    When used as convolution kernel it will compute local mean weighted with PDF of normal distribution
    """
    if size % 2 == 0:
        raise ValueError("Size must be odd number for Gaussian blur kernel!")

    def local_meshgrid(s: int):
        """Returns local meshgrid with given size centred at zero using Manhattan distance"""
        center = s // 2
        return np.abs(np.arange(s) - center) + np.abs(np.arange(s).reshape(-1, 1) - center)

    gaussian = stats.multivariate_normal(np.zeros(2), sigma * np.eye(2))  # 2D Gaussian
    gaussian = gaussian.pdf(np.dstack([local_meshgrid(size), local_meshgrid(size)]))  # compute PDF for local meshgrid

    kernel = np.ones([size, size]) * gaussian
    return kernel / kernel.sum()  # normalize kernel
