# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from LinearRegression.RidgeRegression.base.baseRidgeRegression import BaseRidgeRegression


class SVD(BaseRidgeRegression):

    # https://it.wikipedia.org/wiki/Regolarizzazione_di_Tichonov#Collegamenti_con_la_decomposizione_ai_valori_singolari_e_il_filtro_di_Wiener
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        y = y.reshape(-1, 1)

        # Singular Value Decomposition.
        # U: Unitary array; eigenvectors of SSᵀ
        # Σ: Vector with the singular values, sorted in descending order
        # Vᵀ: Unitary array (Conjugate transpose); eigenvectors of SᵀS
        U, Σ, Vᵀ = numpy.linalg.svd(S, full_matrices=False)

        # numpy.diag: Extract a diagonal
        # w = V·diag(Σ/(Σ² + ɑ²))·Uᵀ·y
        w = Vᵀ.T.dot(numpy.diag(Σ / (Σ ** 2 + self.alpha ** 2))).dot(U.T.dot(y))

        return w.reshape(1, -1)
