# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy


class SVD:

    # https://it.wikipedia.org/wiki/Regolarizzazione_di_Tichonov#Collegamenti_con_la_decomposizione_ai_valori_singolari_e_il_filtro_di_Wiener
    def elaborate(self, S: numpy.ndarray, y: numpy.ndarray, alpha: float) -> None:
        y = y.reshape((16512, 1))
        S = numpy.append(numpy.ones(S.shape[0]).reshape(-1, 1), S, axis=1)

        # Singular Value Decomposition.
        # U: Unitary array; eigenvectors of SSᵀ
        # Σ: Vector with the singular values, sorted in descending order
        # Vᵀ: Unitary array (Conjugate transpose); eigenvectors of SᵀS
        U, Σ, Vᵀ = numpy.linalg.svd(S, full_matrices=False)
        UR = numpy.dot(U.T, y)

        # Expand alpha to a collection if it's just a single value
        alpha = numpy.ones(y.shape[1]) * alpha

        # Normalize alpha by the LSV norm
        norm = Σ[0]
        normalized_alpha = alpha * norm

        # Compute weights for each alpha
        # Returns the sorted unique elements of an array.
        unique_alphas = numpy.unique(normalized_alpha)
        wt = numpy.zeros((S.shape[1], y.shape[1]))

        for ua in unique_alphas:
            selvox = numpy.nonzero(normalized_alpha == ua)[0]
            awt = Vᵀ.T.dot(numpy.diag(Σ / (Σ ** 2 + ua ** 2))).dot(UR[:, selvox])
            wt[:, selvox] = awt

        self.w = wt[:, 0]

    def predict(self, x_test: numpy.ndarray) -> numpy.ndarray:
        m = x_test.shape[0]

        if self.w.shape[0] > x_test.shape[1]:
            x_test = numpy.append(numpy.ones(m).reshape(-1, 1), x_test, axis=1)

        y_predict = (self.w * x_test).sum(axis=1)

        return y_predict
