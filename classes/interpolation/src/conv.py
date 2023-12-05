import numpy as np
from numpy.typing import NDArray


def conv1d_interpolate(x_measure: NDArray, y_measure: NDArray, x_interpolate: NDArray, kernel: callable) -> NDArray:
    """
    Interpolate using a convolution kernel

    :param x_measure: x values of the measurements
    :param y_measure: y values of the measurements
    :param x_interpolate: x values of the interpolation
    :param kernel: callable interpolation kernel accepting x, offset and width

    :return: y values of the interpolation
    """
    width = x_measure[1] - x_measure[0]  # store period between samples
    kernels = np.asarray([kernel(x_interpolate, offset=offset, width=width) for offset in x_measure])

    return y_measure @ kernels


def image_interpolate(image: NDArray, kernel: callable, ratio: int) -> NDArray:
    """Interpolate an image using a convolution kernel"""

    def row_column_interpolate(row: NDArray) -> NDArray:
        """Interpolate a single row or column of the image"""
        x_measure = np.arange(len(row))
        x_interpolate = np.linspace(0, len(row), ratio * len(row), endpoint=False)
        return conv1d_interpolate(x_measure, row, x_interpolate, kernel)

    interpolated = np.apply_along_axis(row_column_interpolate, 1, image)
    return np.apply_along_axis(row_column_interpolate, 0, interpolated)


def rgb_image_interpolate(image: NDArray, kernel: callable, ratio: int) -> NDArray:
    """Interpolate an RGB image by applying the image interpolation function to each channel"""
    return np.stack([image_interpolate(image[:, :, channel], kernel, ratio) for channel in range(3)], axis=2)
