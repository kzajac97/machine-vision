import numpy as np
from numpy.typing import NDArray


def apply_rgb(func: callable, image: NDArray, *args, **kwargs) -> NDArray:
    """
    Applies a function to each color channel of an image.

    :param func: function to apply to each color channel
    :param image: image to apply function to

    :return: image after function has been applied to each color channel
    """
    return np.dstack([func(image[:, :, channel], *args, **kwargs) for channel in range(3)])
