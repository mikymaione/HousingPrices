# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from baseRidgeRegression import BaseRidgeRegression


class Lasso(BaseRidgeRegression):
    algo = 'Lasso'
    num_iters = 100

    # https://en.wikipedia.org/wiki/Lasso_(statistics)
    # https://xavierbourretsicotte.github.io/lasso_implementation.html
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        N = S.shape[0]
        features = S.shape[1]
        β = numpy.ones((features, 1))

        self.λ = self.alpha * N

        for n in range(self.num_iters):
            for j in range(features):
                Xⱼ = S[:, j].reshape(-1, 1)
                y_pred = S @ β

                ρ = Xⱼ.T @ (y - y_pred + β[j] * Xⱼ)
                ρ = ρ.item()

                if j == 0:
                    β[j] = ρ
                else:
                    β[j] = self.Sα(ρ)

        return β.reshape(-1)

    def Sα(self, ρ: float) -> float:
        if ρ < -self.λ:
            return ρ + self.λ
        elif ρ > self.λ:
            return ρ - self.λ
        else:
            return 0
