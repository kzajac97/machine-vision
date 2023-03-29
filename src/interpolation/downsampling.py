import numpy as np
from skimage.util import view_as_windows


def downsample(image: np.array, kernel_size: int = 2) -> np.array:
    """Downsample an image using a convolution kernel"""
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    windows = view_as_windows(image, window_shape=kernel_size, step=kernel_size)
    return np.apply_over_axes(np.sum, windows * kernel, axes=(2, 3)).squeeze()


def nonlinear_downsample(image: np.array, aggregate: callable, kernel_size: int = 2) -> np.array:
    """Downsample an image using any aggregation function, such as np.max or np.median"""
    windows = view_as_windows(image, window_shape=kernel_size, step=kernel_size)
    return np.apply_over_axes(aggregate, windows, axes=(2, 3)).squeeze()


def rgb_downsample(image: np.array, kernel_size: int = 2, downsample_func: callable = downsample) -> np.array:
    """Downsample an RGB image by applying the downsample function to each channel"""
    return np.stack([downsample_func(image[:, :, channel], kernel_size) for channel in range(3)], axis=2)
