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
    num_iters = 100

    # https://en.wikipedia.org/wiki/Lasso_(statistics)
    # https://xavierbourretsicotte.github.io/lasso_implementation.html
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        β = self.coordinate_descent_lasso(S, y, self.alpha)

        return β

    def S(self, ρ: float, λ: float) -> float:
        if ρ < - λ:
            return ρ + λ
        elif ρ > λ:
            return ρ - λ
        else:
            return 0

    def coordinate_descent_lasso(self, X: numpy.ndarray, y: numpy.ndarray, λ: float) -> numpy.ndarray:
        features = X.shape[1]
        β = numpy.ones((features, 1))

        for n in range(self.num_iters):
            for j in range(features):
                Xⱼ = X[:, j].reshape(-1, 1)
                y_ = X @ β

                ρ = Xⱼ.T @ (y - y_ + β[j] * Xⱼ)
                ρ = ρ.item()

                β[j] = self.S(ρ, λ)

        return β.flatten()
