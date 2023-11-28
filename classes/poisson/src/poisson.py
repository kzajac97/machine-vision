import numpy as np
from numpy.typing import NDArray


def poissoning(image: NDArray, lambda_value: float) -> np.ndarray:
    """
    :param image: numpy array of shape (H, W, C) or (H, W)
    :param lambda_value: number of simulated photons per pixel (or per channel, depending on the image)
    """
    noised_image = np.random.poisson(image / image.max() * lambda_value) / lambda_value
    noised_image = np.clip(noised_image * 255, 0, 255).astype(int)
    return noised_image
