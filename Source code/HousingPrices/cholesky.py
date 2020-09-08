# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from baseRidgeRegression import BaseRidgeRegression


class Cholesky(BaseRidgeRegression):
    algo = 'Cholesky'

    # https://it.wikipedia.org/wiki/Regolarizzazione_di_Tichonov#Regolarizzazione_generalizzata_di_Tikhonov
    # https://xavierbourretsicotte.github.io/intro_ridge.html
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        features = S.shape[1]

        I = numpy.eye((features))
        αI = self.alpha * I
        Γ = αI
        ΓᵀΓ = Γ.T @ Γ

        Sᵀy = S.T @ y
        SᵀS = S.T @ S

        # Closed form solution:
        # (Sᵀ·S + Γᵀ·Γ)·w = Sᵀ·y
        # w = (Sᵀ·S + Γᵀ·Γ)⁻¹·Sᵀ·y
        w = numpy.linalg.inv(SᵀS + ΓᵀΓ) @ Sᵀy

        return w.reshape(-1)
