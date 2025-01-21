import numpy as np
from numpy.typing import NDArray

from src.interpolate.core import KernelCallable


def dirac_interpolate(
    x_measure: NDArray, y_measure: NDArray, ratio: int, kernel_size: int, kernel_width: float, kernel: KernelCallable
) -> NDArray:
    """
    :param x_measure: array of samples from x-axis of the measured function, needs to be uniformly spaced
    :param y_measure: array of samples from y-axis of the measured function
    :param ratio: integer interpolation ratio
    :param kernel_size: desired size of the kernel for interpolation
    :param kernel_width: kernel width, needs to be selected according to the kernel used and number of samples
    :param kernel: kernel as callable numpy function
    """
    if not isinstance(ratio, int):
        raise ValueError("Interpolation ratio must be an integer!")

    # create new interpolation x-axis and y-axis
    # those have length ratio * len(x_measure) + kernel_size to account for decrease in size from valid convolution
    x_interpolate = np.linspace(x_measure[0], x_measure[-1], ratio * len(x_measure))
    y_interpolate = np.zeros(len(x_interpolate))
    # compute kernel ratio and prefill y_interpolate with measured values
    # this creates a representation of the original function as dirac delta functions in the target domain,
    # where x-axis is the size expected after interpolation
    y_interpolate[::ratio] = y_measure
    # create centred kernel on -1 to 1 range with given width
    x_kernel = np.linspace(-1, 1, kernel_size)
    y_kernel = kernel(x_kernel, offset=0, width=kernel_width)
    # return valid convolution, which will contain interpolation
    return np.convolve(y_interpolate, y_kernel, mode="same")
