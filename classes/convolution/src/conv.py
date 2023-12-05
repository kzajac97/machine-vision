import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray
from scipy.ndimage import convolve


def _convolve(image: NDArray, kernel: NDArray, stride: int) -> NDArray:
    """Convolve implementation for greyscale images with stride"""
    if stride == 1:
        return convolve(image, kernel)  # shorthand for convolve without stride

    input_width, image_height = image.shape
    kernel_width, kernel_height = kernel.shape
    output_shape = ((input_width - kernel_width) // stride + 1, (image_height - kernel_height) // stride + 1)

    # create a view of the input array with the desired strides
    strides = (stride * image.strides[0], stride * image.strides[1]) + image.strides
    strided_input = as_strided(image, shape=output_shape + (kernel_width, kernel_height), strides=strides)
    return np.einsum("ijkl,kl->ij", strided_input, kernel)


def convolve_with_stride(image: NDArray, kernel: NDArray, stride: int = 1, padding: int = 0) -> NDArray:
    """
    Convolve an image with a kernel using a given stride.

    :param image: image to convolve given as numpy array
    :param kernel: kernel to convolve image with given as numpy array
    :param stride: stride to use for convolution, given in pixels
    :param padding: padding given in pixels, which will be added around the image
    """
    if padding is not None:
        image = np.pad(image, padding, mode="constant")

    if image.ndim == 3:  # RGB image
        return np.dstack([_convolve(image[:, :, channel], kernel, stride) for channel in range(3)])

    if image.ndim == 2:
        return _convolve(image, kernel, stride)  # greyscale image

    raise ValueError(f"Image shape {image.shape} not supported")
