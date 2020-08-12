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

    def elaborate(self, S: numpy.ndarray, y: numpy.ndarray, alpha: float) -> None:
        features = S.shape[1]
        y_ = y.reshape(-1, 1)

        S_offset = numpy.average(S, axis=0)
        y_offset = numpy.average(y_, axis=0)

        S_ = S - S_offset
        y_ = y_ - y_offset

        # Normalization is the process of scaling individual samples to have unit norm.
        # This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
        # This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
        # The function normalize provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms.
        S_, S_scale = preprocessing.normalize(S_, axis=0, copy=False, return_norm=True)

        Sy = S_.T.dot(y_)
        A = S_.T.dot(S_)

        for i in range(features):
            A[i, i] += alpha

        w = numpy.linalg.solve(A, Sy).T
        self.w = w / S_scale
        self.intercetta = y_offset - S_offset.dot(self.w.T)

    def predict(self, x_test: numpy.ndarray) -> numpy.ndarray:
        y_predict = x_test.dot(self.w.T) + self.intercetta

        return y_predict.reshape(1, -1)
