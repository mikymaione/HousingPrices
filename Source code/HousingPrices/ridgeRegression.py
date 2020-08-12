# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy
from sklearn import preprocessing


class RidgeRegression:

    def elaborate(self, S: numpy.ndarray, y: numpy.ndarray, ɑ: float) -> None:
        features = S.shape[1]
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

        Sᵀy = S.T.dot(y)
        SᵀS = S.T.dot(S)

        # ɑ·I
        for i in range(features):
            SᵀS[i, i] += ɑ

        # Solve a linear matrix equation, or system of linear scalar equations.
        # Computes the “exact” solution, w, of the well-determined, i.e., full rank, linear matrix equation Sᵀ·S·w = Sᵀ·y
        w = numpy.linalg.solve(SᵀS, Sᵀy).T

        self.w = w / norm_L2
        self.intercetta = y_wam - S_wam.dot(self.w.T)

    def predict(self, x_test: numpy.ndarray) -> numpy.ndarray:
        y_predict = x_test.dot(self.w.T) + self.intercetta

        return y_predict.reshape(1, -1)