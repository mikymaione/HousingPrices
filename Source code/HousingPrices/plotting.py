# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import List, Tuple

from dataTypes import DataElaboration


class Plotting:

    @staticmethod
    def plot_DataElaboration(titolo: str, labels: List[str], P: DataElaboration) -> None:
        # plt.style.use('grayscale')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(titolo)
        fig.canvas.set_window_title(titolo)

        ax1.set_title("MAPE")
        ax2.set_title("R²")

        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("MAPE")

        ax2.set_xlabel("Alpha")
        ax2.set_ylabel("R²")

        ax1.plot(P.alphas, P.mapes_cholesky, label=labels[0])
        ax1.plot(P.alphas, P.mapes_svd, label=labels[1])
        ax1.plot(P.alphas, P.mapes_lsqr, label=labels[2])

        ax2.plot(P.alphas, P.r2s_cholesky, label=labels[0])
        ax2.plot(P.alphas, P.r2s_svd, label=labels[1])
        ax2.plot(P.alphas, P.r2s_lsqr, label=labels[2])

        ax1.legend()
        ax2.legend()
        plt.show()

    @staticmethod
    def scatterPlot(plotTitle: str, y_predict: numpy.ndarray, y_test: numpy.ndarray, figsize: Tuple[float, float] = None) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(plotTitle)
        fig.canvas.set_window_title(plotTitle)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")

        ax.scatter(y_predict, y_test, alpha=0.5)

        line = mlines.Line2D([0, 1], [0, 1], color='red')
        transform = ax.transAxes
        line.set_transform(transform)

        ax.add_line(line)

        plt.show()
