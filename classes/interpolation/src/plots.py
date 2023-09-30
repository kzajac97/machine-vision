from typing import NamedTuple, Sequence

import matplotlib.pyplot as plt


class PlotData(NamedTuple):
    """Simple container for plot data"""

    x: Sequence[float]
    y: Sequence[float]


def plot_interpolation(measure: PlotData, interpolation: PlotData, real: PlotData, figsize=(12, 8)):
    """Plot the interpolation, measurements and the real function"""
    figure = plt.figure(figsize=figsize)

    plt.scatter(measure.x, measure.y, label="Measurements")
    plt.plot(interpolation.x, interpolation.y, label="Interpolation")
    plt.plot(real.x, real.y, label="Real Function")

    plt.legend()
