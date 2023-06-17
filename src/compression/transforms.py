from abc import abstractmethod
from typing import Optional

import numpy as np
import pywt


class CompressionTransform:
    """
    Interface for compression transforms.
    """

    @abstractmethod
    def forward(self, variables: np.array) -> np.array:
        ...

    @abstractmethod
    def backward(self, variables: np.array) -> np.array:
        ...


class FourierTransform2D(CompressionTransform):
    """
    2D Fourier transform used for compression.
    Inverse transform uses absolute value by default.
    """

    def forward(self, variables: np.array) -> np.array:
        return np.fft.fft2(variables)

    def backward(self, variables: np.array) -> np.array:
        return np.abs(np.fft.ifft2(variables))


class WaveletTransform2D(CompressionTransform):
    """
    2D wavelet transform used for compression.
    """

    def __init__(self, wavelet_name: str, level: int):
        self.wavelet_name = wavelet_name
        self.level = level
        self.slices: Optional[np.array] = None

    def forward(self, variables: np.array) -> np.array:
        transformed = pywt.wavedec2(variables, self.wavelet_name, level=self.level)
        coefficients, slices = pywt.coeffs_to_array(transformed)
        self.slices = slices

        return coefficients

    def backward(self, variables: np.array) -> np.array:
        if self.slices is None:
            raise ValueError("Cannot perform inverse transform without first performing forward transform!")

        variables = pywt.array_to_coeffs(variables, self.slices, output_format="wavedec2")
        return pywt.waverec2(variables, self.wavelet_name)


def compress_and_decompress(image: np.array, transform: CompressionTransform, compression: float) -> np.array:
    """
    Compresses and decompresses an image using the Fourier transform.
    This function can be used to see compression and decompression effects.

    :param image: greyscale image
    :param transform: transform to use, using CompressionTransform interface
    :param compression: ratio of coefficients to remove

    :return: image after compression and decompression
    """
    transformed = transform.forward(image)
    coefficients = np.sort(np.abs(transformed.reshape(-1)))  # sort by magnitude

    threshold = coefficients[int(compression * len(coefficients))]
    indices = np.abs(transformed) > threshold

    decompressed = transformed * indices
    return transform.backward(decompressed)
