import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray


def convolve_with_stride(image: NDArray, kernel: NDArray, stride: int, padding: int | None = None) -> NDArray:
    """
    Convolve an image with a kernel using a given stride.

    :param image: image to convolve given as numpy array
    :param kernel: kernel to convolve image with given as numpy array
    :param stride: stride to use for convolution, given in pixels
    :param padding: padding given in pixels, which will be added around the image
    """
    if padding is not None:
        image = np.pad(image, padding, mode="constant")

    # get the dimensions of the input array and the kernel
    input_shape = image.shape
    kernel_shape = kernel.shape
    # calculate the shape of the output array
    output_shape = ((input_shape[0] - kernel_shape[0]) // stride + 1, (input_shape[1] - kernel_shape[1]) // stride + 1)

    # create a view of the input array with the desired strides
    strides = (stride * image.strides[0], stride * image.strides[1]) + image.strides
    strided_input = as_strided(image, shape=output_shape + kernel_shape, strides=strides)
    return np.einsum("ijkl,kl->ij", strided_input, kernel)
