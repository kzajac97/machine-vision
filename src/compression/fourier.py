import numpy as np


def compress_and_decompress(image: np.array, compression: float = 0.9):
    """
    Compresses and decompresses an image using the Fourier transform.
    This function can be used to see compression and decompression effects.

    :param image: greyscale image
    :param compression: ratio of coefficients to remove

    :return: image after compression and decompression
    """
    transformed = np.fft.fft2(image)
    coeffs = np.sort(np.abs(transformed.reshape(-1)))  # sort by magnitude

    threshold = coeffs[int(compression * len(coeffs))]
    indices = np.abs(transformed) > threshold

    decompressed = transformed * indices
    return np.abs(np.fft.ifft2(decompressed))
