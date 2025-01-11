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


def create_grid(limits: tuple[int, int], shape: tuple[int, int]) -> NDArray:
    """
    Creates grid of points, which are indexing the image.

    :param limits: maximum row and column values, the image size before interpolation
    :param shape: image shape, for interpolation-grid image_shape is different from limits
    """
    max_row_value, max_column_value = limits
    height, width = shape

    x = np.linspace(1, width, max_row_value, endpoint=True)
    y = np.linspace(1, height, max_column_value, endpoint=True)
    xx, yy = np.meshgrid(x, y)
    return np.vstack([yy.ravel(), xx.ravel()]).T


def image_interpolate2d(image: NDArray, kernel: KernelCallable, ratio: int, eps: float = float(0)) -> NDArray:
    """
    Interpolate image using 2D kernel interpolation

    :param image: grayscale image to interpolate as 2D NDArray
    :param kernel: Callable interpolation kernel accepting 2D grid, offset and width
    :param ratio: up-scaling factor
    :param eps: width correction coefficient to avoid overlapping kernels (use for sinc and keys kernels)

    :return: interpolated image as 2D NDArray
    """
    target_shape = np.asarray(image.shape) * ratio
    # create indexing for image starting from 1 as array of 2D points (shape: [H*W, 2])
    # [1,1], [1,2], [1,3], ..., [2,1], [2,2], [2,3], ... [H,W]
    image_grid = create_grid(image.shape, image.shape)  # type: ignore
    # create indexing for interpolation grid, filling points in-between points from image grid
    # [1,1], [1, 1 + 1/ratio], [1, 1 + 2/ratio], ..., [H, W]
    interpolate_grid = create_grid(target_shape, image.shape)  # type: ignore

    interpolated = np.zeros(target_shape)  # do not store all kernels to save memory in 2D
    for point, value in zip(image_grid, image.ravel()):
        kernel_value = value * kernel(interpolate_grid, offset=point, width=(1 - eps))  # type: ignore
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
