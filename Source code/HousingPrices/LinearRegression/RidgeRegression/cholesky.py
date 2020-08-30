# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from LinearRegression.RidgeRegression.base.baseRidgeRegression import BaseRidgeRegression


class Cholesky(BaseRidgeRegression):

    # https://it.wikipedia.org/wiki/Regolarizzazione_di_Tichonov#Regolarizzazione_generalizzata_di_Tikhonov
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        features = S.shape[1]

        Sᵀy = S.T.dot(y)
        SᵀS = S.T.dot(S)

        for i in range(features):
            SᵀS[i, i] += self.alpha

        # Solve a linear matrix equation, or system of linear scalar equations.
        # Computes the “exact” solution, w, of the well-determined, i.e., full rank, linear matrix equation Sᵀ·S·w = Sᵀ·y
        w = numpy.linalg.solve(SᵀS, Sᵀy).T

        return w
