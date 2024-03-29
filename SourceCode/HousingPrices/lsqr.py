# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy
import scipy.sparse.linalg

from baseRidgeRegression import BaseRidgeRegression


class LSQR(BaseRidgeRegression):
    algo = 'LSQR'
    
    # https://it.wikipedia.org/wiki/Algoritmo_di_Levenberg-Marquardt
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        # Find the least-squares solution to a large, sparse, linear system of equations.
        # Levenberg–Marquardt algorithm also known as the damped least-squares
        # [Sᵀ·S + √α·diag(Sᵀ·S)]·α = Sᵀ·[y - w]
        w = scipy.sparse.linalg.lsqr(S, y, damp=self.alpha ** 0.5)

        return w[0]
