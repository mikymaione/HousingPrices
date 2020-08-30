# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from sklearn import preprocessing

from Utility.dataManager import DataManager
from Utility.dataTypes import ElaborationResult


class BaseRidgeRegression:
    w: numpy.ndarray
    intercetta: numpy.ndarray

    def __init__(self, ɑ: float):
        self.ɑ = ɑ

    def executeAll(self, S: numpy.ndarray, y: numpy.ndarray, x_test: numpy.ndarray, y_test: numpy.ndarray) -> ElaborationResult:
        self.fit(S=S, y=y)
        R = ElaborationResult(self.w, self.predict(x_test))

        R.mape = DataManager.mean_absolute_percentage_error(y_test=y_test, y_predict=R.y_predict)
        R.r2 = DataManager.coefficient_of_determination(y_test=y_test, y_predict=R.y_predict)

        return R

    def score(self, x_test: numpy.ndarray, y_test: numpy.ndarray) -> numpy.float64:
        y_predict = self.predict(x_test)

        return DataManager.coefficient_of_determination(y_test=y_test, y_predict=y_predict)

    def fit(self, S: numpy.ndarray, y: numpy.ndarray) -> None:
        y = y.reshape(-1, 1)

        # Compute the weighted arithmetic mean along the specified axis.
        S_wam = numpy.average(S, axis=0)
        y_wam = numpy.average(y, axis=0)

        S = S - S_wam
        y = y - y_wam

        # Normalization is the process of scaling individual samples to have unit norm.
        # This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
        # This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
        # The function normalize provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms.
        S, norm_L2 = preprocessing.normalize(X=S, norm="l2", axis=0, copy=False, return_norm=True)

        self.w = self.calculateWeights(S, y) / norm_L2
        self.intercetta = y_wam - S_wam.dot(self.w.T)

    def predict(self, x_test: numpy.ndarray) -> numpy.ndarray:
        y_predict = x_test.dot(self.w.T) + self.intercetta

        return y_predict.reshape(1, -1)

    # abstract
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("Please Implement this method")
