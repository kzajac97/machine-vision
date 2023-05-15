import numpy as np


def poissoning(image: np.array, lambda_value: float) -> np.ndarray:
    """
    :param image: numpy array of shape (H, W, C) or (H, W)
    :param lambda_value: number of simulated photons per pixel (or per channel, depending on the image)
    """
    noised_image = np.random.poisson(image / image.max() * lambda_value) / lambda_value
    noised_image = np.clip(noised_image * 255, 0, 255).astype(int)
    return noised_image
