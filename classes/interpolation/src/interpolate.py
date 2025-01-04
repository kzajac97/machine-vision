from typing import Callable

import numpy as np
from numpy.typing import NDArray

KernelCallable = Callable[[NDArray, NDArray | float, float], NDArray]
InterpolateCallable = Callable[[NDArray, KernelCallable, int], NDArray]


def convolve_interpolate(
    x_measure: NDArray, y_measure: NDArray, x_interpolate: NDArray, kernel: KernelCallable, return_kernels: bool = False
) -> NDArray | tuple[NDArray, NDArray]:
    width = x_measure[1] - x_measure[0]  # store period between samples
    kernels = []

    for x_sample, y_sample in zip(x_measure, y_measure):
        # compute kernel for each sample using convolution, y_sample is just a single value
        kernel = np.convolve(y_sample, kernel(x_interpolate, offset=x_sample, width=width), mode="same")  # type: ignore
        kernels.append(kernel)

    y_interp = np.sum(np.asarray(kernels), axis=0)
    if return_kernels:
        return y_interp, np.asarray(kernels)

    return y_interp


def product_interpolate(
    x_measure: NDArray, y_measure: NDArray, x_interpolate: NDArray, kernel: KernelCallable
) -> NDArray:
    """
    Interpolate using a convolution kernel and efficient dot product

    :param x_measure: x values of the measurements
    :param y_measure: y values of the measurements
    :param x_interpolate: x values of the interpolation
    :param kernel: callable interpolation kernel accepting x, offset and width

    :return: y values of the interpolation
    """
    width = x_measure[1] - x_measure[0]  # store period between samples
    kernels = np.asarray([kernel(x_interpolate, offset=offset, width=width) for offset in x_measure])  # type: ignore

    return y_measure @ kernels


def image_interpolate1d(image: NDArray, kernel: KernelCallable, ratio: int) -> NDArray:
    """Interpolate an image using row and column-wise interpolations"""

    def row_column_interpolate(row: NDArray) -> NDArray:
        """Interpolate a single row or column of the image"""
        x_measure = np.arange(len(row))
        x_interpolate = np.linspace(0, len(row), ratio * len(row), endpoint=False)
        return product_interpolate(x_measure, row, x_interpolate, kernel)

    interpolated = np.apply_along_axis(row_column_interpolate, 1, image)
    return np.apply_along_axis(row_column_interpolate, 0, interpolated)


def _image_grid(shape: tuple[int, int], spacing: float = float(1)) -> NDArray:
    """
    Create a grid of x and y axes for an image. Given `spacing` different from 1 will create interpolation grid.
    Grid is returns as a 2D array with shape (n, 2) where n is the number of points in the grid.
    """
    height, width = shape
    x = np.arange(0, width, spacing)
    y = np.arange(0, height, spacing)
    xx, yy = np.meshgrid(x, y)
    return np.vstack([yy.ravel(), xx.ravel()]).T


def image_interpolate2d(image: NDArray, kernel: KernelCallable, ratio: int) -> NDArray:
    """
    Interpolate image using 2D kernel interpolation

    :param image: grayscale image to interpolate as 2D NDArray
    :param kernel: Callable interpolation kernel accepting 2D grid, offset and width
    :param ratio: up-scaling factor

    :return: interpolated image as 2D NDArray
    """
    target_shape = np.asarray(image.shape) * ratio
    # create grid of 2D points, which are indexing the image
    # [0,0], [0,1], [0,2], ... [1,0], [1,1], [1,2], ... [W,H]
    image_grid = _image_grid(image.shape, spacing=1)  # type: ignore
    # create grid of 2D points, which are indexing the interpolated image with the same range as original image
    # [0,0], [0,1/ratio], [0,2/ratio], ... [1/ratio,0], [1/ratio,1/ratio], [1/ratio,2/ratio], ... [W,H]
    interpolate_grid = _image_grid(image.shape, spacing=1 / ratio)  # type: ignore

    interpolated = np.zeros(target_shape)  # do not store all kernels to save memory in 2D
    for point, value in zip(image_grid, image.ravel()):
        kernel_value = value * kernel(interpolate_grid, offset=point, width=1)  # type: ignore
        interpolated += kernel_value.reshape(target_shape)

    return interpolated


def rgb_image_interpolate(
    image: NDArray, kernel: KernelCallable, ratio: int, interpolate: InterpolateCallable = image_interpolate1d
) -> NDArray:
    """
    Interpolate an RGB image by applying the image interpolation function to each channel

    :param image: RGB image to interpolate as 3D NDArray
    :param kernel: Callable interpolation kernel accepting x, offset and width
    :param ratio: up-scaling factor
    :param interpolate: image interpolation function to use, default is 1D interpolation
    """
    return np.stack([interpolate(image[:, :, channel], kernel, ratio) for channel in range(3)], axis=2)
