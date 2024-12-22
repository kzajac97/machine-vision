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
