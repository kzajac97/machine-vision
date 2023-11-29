from abc import abstractmethod
from typing import Optional

import numpy as np
import pywt
from numpy.typing import NDArray


class CompressionTransform:
    """
    Interface for compression transforms.
    """

    @abstractmethod
    def forward(self, variables: NDArray) -> NDArray:
        ...

    @abstractmethod
    def backward(self, variables: NDArray) -> NDArray:
        ...


class FourierTransform2D(CompressionTransform):
    """
    2D Fourier transform used for compression.
    Inverse transform uses absolute value by default.
    """

    def forward(self, variables: NDArray) -> NDArray:
        return np.fft.fft2(variables)

    def backward(self, variables: NDArray) -> NDArray:
        return np.abs(np.fft.ifft2(variables))


class WaveletTransform2D(CompressionTransform):
    """
    2D wavelet transform used for compression.
    """

    def __init__(self, wavelet_name: str, level: int):
        self.wavelet_name = wavelet_name
        self.level = level
        self.slices: Optional[NDArray] = None

    def forward(self, variables: NDArray) -> NDArray:
        transformed = pywt.wavedec2(variables, self.wavelet_name, level=self.level)
        coefficients, slices = pywt.coeffs_to_array(transformed)
        self.slices = slices

        return coefficients

    def backward(self, variables: NDArray) -> NDArray:
        if self.slices is None:
            raise ValueError("Cannot perform inverse transform without first performing forward transform!")

        variables = pywt.array_to_coeffs(variables, self.slices, output_format="wavedec2")  # type: ignore
        return pywt.waverec2(variables, self.wavelet_name)


def compress_and_decompress(image: NDArray, transform: CompressionTransform, compression: float) -> NDArray:
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
