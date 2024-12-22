from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _full_conv1d(signal: NDArray, kernel: NDArray, step: int = 1, padding: int = 0):
    padded = np.pad(signal, padding)  # zero-pad on both sides
    # compute aligned length of output signal
    # ((x + p) + m - 1) / s  (m = len(kernel), p = padding, s = step)
    output_size = len(padded) + len(kernel) - 1
    output_size = int(np.ceil(output_size / step))
    output = np.zeros(output_size)

    # n = tracking index(for step = 1, equal to ((x + p) + m - 1) / s
    # y = output, x = signal, k = kernel, m = len(kernel)

    # y[0] = x[0]k[0]
    # y[1] = x[0]k[1] + x[1]k[0]
    # y[2] = x[0]k[2] + x[1]k[1] + x[2]k[0]
    # ...
    # y[m] = x[0]k[m] + x[1]k[m-1] + ... + x[m]k[0]  # first full kernel overlap (for n == m this happens once)
    # ...
    # y[n] = x[n]k[m] + x[1]k[m-1] + ... + x[n+m]k[0]  # full overlap when n > m
    # ...
    # y[n + m - 1] = x[n] k[m]  # last output element
    for conv_step in range(output_size):
        for kernel_step in range(len(kernel)):
            index = conv_step * step - kernel_step  # note: tracking index can be negative here
            if 0 <= index < len(signal):  # index needs to be within bounds of signal
                # increment output position by summing aligned signal and kernel
                output[conv_step] += signal[index] * kernel[kernel_step]

    return np.asarray(output)


def _valid_conv1d(signal: NDArray, kernel: NDArray, step: int = 1, padding: int = 0) -> NDArray:
    # reverse the kernel -> needed for simplified implementation to agree with mathematical definition
    kernel = kernel[::-1]
    padded = np.pad(signal, padding)  # zero-pad on both sides
    kernel_size = len(kernel)
    # compute aligned length of output signal
    # ((x + p) - m + 1) / s  (m = len(kernel), p = padding, s = step)
    output_size = len(padded) - len(kernel) + 1
    output_size = int(np.ceil(output_size / step))
    output = np.zeros(output_size)

    for conv_step, index in enumerate(range(0, len(signal), step)):
        if index + kernel_size <= len(signal):  # needed, when len(single) is not multiple of kernel_size or step
            # sum of aligned element-wise multiplication of signal and kernel
            # x is the signal, k is the kernel, n is tracking index and m = len(kernel)
            # x[n] + k[n] + x[n+1] + k[n+1] + ... + x[n+m] + k[m]
            output[conv_step] = np.sum(signal[index : index + kernel_size] * kernel)

    return output


def _full_conv2d(image: NDArray, kernel: NDArray, step: int = 1, padding: int = 0) -> NDArray:
    raise NotImplementedError("2D convolution is not implemented yet!")


def _valid_conv2d(image: NDArray, kernel: NDArray, step: int = 1, padding: int = 0) -> NDArray:
    padded = np.pad(image, padding)
    # reverse the kernel in both axes -> needed for simplified implementation to agree with mathematical definition
    kernel = np.flipud(np.fliplr(kernel))

    height, width = padded.shape
    kernel_height, kernel_width = kernel.shape
    # compute aligned length of output signal, where x is either height or width of the image
    # ((x + p) - m + 1) / s  (m = len(kernel), p = padding, s = step)
    output_height = height - kernel_height + 1
    output_width = width - kernel_width + 1
    output_height = int(np.ceil(output_height / step))
    output_width = int(np.ceil(output_width / step))

    if output_height <= 0 or output_width <= 0:  # safety check
        raise ValueError("Kernel size is too large for valid convolution")

    output = np.zeros((output_height, output_width))

    for conv_h_step, h_index in enumerate(range(0, output_height, step)):
        for conv_w_step, w_index in enumerate(range(0, output_width, step)):
            # slice image to kernel sized window (kw and kh denotes kernel width and height)
            # sum of aligned element-wise multiplication of signal and kernel

            # x is the signal, k is the kernel, h is tracking index for height, w is tracking index for width
            # x[h+kh, w+kw] * k[0, 0] + x[h+kh, w+kw+1] * k[0, 1] + ... + x[h+kh, w+kw+kw] * k[0, kw]
            window = image[h_index: h_index + kernel_height, w_index: w_index + kernel_width]
            output[conv_h_step, conv_w_step] = np.sum(window * kernel)

    return output


def convolve(
    signal: NDArray, kernel: NDArray, step: int = 1, padding: int = 0, mode: Literal["full", "valid"] = "full"
) -> NDArray:
    """
    Educational implementation of `np.convolve` for 1D or 2D signals and kernels with step and padding support.
    See: https://numpy.org/doc/2.1/reference/generated/numpy.convolve.html for more.

    :param signal: Supports 1D and 2D signals
    :param kernel: Kernel to convolve with the signal, must have the same dimensionality as the signal
    :param step: step size for the convolution
    :param padding: input padding for the convolution, applied to signal on both sides
    :param mode: "full" or "valid" mode for the convolution:
                 "full" returns the convolution at each point of overlap
                 "valid" returns convolution, only for points where the signals overlap completely
    """
    if signal.ndim != kernel.ndim:
        raise ValueError("Signal and kernel must have the same dimensionality!")

    dim = signal.ndim

    match mode, dim:
        case "full", 1:
            return _full_conv1d(signal, kernel, step, padding)
        case "valid", 1:
            return _valid_conv1d(signal, kernel, step, padding)
        case "full", 2:
            ...
        case "valid", 2:
            ...
        case _, _:
            raise ValueError(f"Unknown mode: {mode}")
