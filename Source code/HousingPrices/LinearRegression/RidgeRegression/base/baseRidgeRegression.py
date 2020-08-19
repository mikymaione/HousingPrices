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

from sklearn import preprocessing

from Utility.dataUtility import DataUtility


class BaseRidgeRegression:

    def executeAll(self, S: numpy.ndarray, y: numpy.ndarray, ɑ: float, normalize: bool, x_test: numpy.ndarray,
                   y_test: numpy.ndarray, showPlot: bool = False, plotTitle: str = ""):
        self.elaborate(S=S, y=y, ɑ=ɑ, normalize=normalize)
        y_predict = self.predict(x_test)

        mape = DataUtility.mean_absolute_percentage_error(y_test=y_test, y_predict=y_predict)
        r2 = DataUtility.coefficient_of_determination(y_test=y_test, y_predict=y_predict)

        if showPlot:
            fig, ax = plt.subplots()
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

        return mape, r2

    def elaborate(self, S: numpy.ndarray, y: numpy.ndarray, ɑ: float, normalize: bool) -> None:
        norm_L2 = 1
        y = y.reshape(-1, 1)

        # Compute the weighted arithmetic mean along the specified axis.
        S_wam = numpy.average(S, axis=0)
        y_wam = numpy.average(y, axis=0)

        S = S - S_wam
        y = y - y_wam

        if normalize:
            # Normalization is the process of scaling individual samples to have unit norm.
            # This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
            # This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
            # The function normalize provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms.
            S, norm_L2 = preprocessing.normalize(X=S, norm="l2", axis=0, copy=False, return_norm=True)

        self.w = self.calculateWeights(S, y, ɑ) / norm_L2
        self.intercetta = y_wam - S_wam.dot(self.w.T)

    def predict(self, x_test: numpy.ndarray) -> numpy.ndarray:
        y_predict = x_test.dot(self.w.T) + self.intercetta

        return y_predict.reshape(1, -1)

    # abstract
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray, ɑ: float) -> numpy.ndarray:
        raise NotImplementedError("Please Implement this method")
