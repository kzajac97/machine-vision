import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray


def convolve_with_stride(image: NDArray, kernel: NDArray, stride: int) -> NDArray:
    """Convolve an image with a kernel using a given stride."""
    # get the dimensions of the input array and the kernel
    input_shape = image.shape
    kernel_shape = kernel.shape
    # calculate the shape of the output array
    output_shape = ((input_shape[0] - kernel_shape[0]) // stride + 1, (input_shape[1] - kernel_shape[1]) // stride + 1)

    # create a view of the input array with the desired strides
    strides = (stride * image.strides[0], stride * image.strides[1]) + image.strides
    strided_input = as_strided(image, shape=output_shape + kernel_shape, strides=strides)
    return np.einsum("ijkl,kl->ij", strided_input, kernel)


def conv1d_interpolate(x_measure: np.array, y_measure: np.array, x_interpolate: np.array, kernel: callable) -> np.array:
    """
    Interpolate using a convolution kernel

    :param x_measure: x values of the measurements
    :param y_measure: y values of the measurements
    :param x_interpolate: x values of the interpolation
    :param kernel: callable interpolation kernel accepting x, offset and width

    :return: y values of the interpolation
    """
    width = x_measure[1] - x_measure[0]  # store period between samples
    kernels = [kernel(x_interpolate, offset=offset, width=width) for offset in x_measure]

    return y_measure @ kernels


def image_interpolate(image: np.array, kernel: callable, ratio: int) -> np.array:
    """Interpolate an image using a convolution kernel"""

    def row_column_interpolate(row: np.array) -> np.array:
        """Interpolate a single row or column of the image"""
        x_measure = np.arange(len(row))
        x_interpolate = np.linspace(0, len(row), ratio * len(row), endpoint=False)
        return conv1d_interpolate(x_measure, row, x_interpolate, kernel)

    interpolated = np.apply_along_axis(row_column_interpolate, 1, image)
    return np.apply_along_axis(row_column_interpolate, 0, interpolated)


def rgb_image_interpolate(image: np.array, kernel: callable, ratio: int) -> np.array:
    """Interpolate an RGB image by applying the image interpolation function to each channel"""
    return np.stack([image_interpolate(image[:, :, channel], kernel, ratio) for channel in range(3)], axis=2)
