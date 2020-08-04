# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Ridge regression algorithm for regression with square loss.

import numpy as np


class RidgeRegression:

    def Elaborate(self, S, y, α):
        S_t = np.transpose(S)
        S_t_S = np.dot(S_t, S)
        inv = np.linalg.inv(S_t_S)
        w = np.dot(inv, S_t).dot(y)

        return w

    def ridge(self, S, R, α):
        """Uses ridge regression to find a linear transformation of [S] that approximates [R]. The regularization parameter is [α].

        Parameters
        ----------
        S : array_like, shape (T, N)
            Stimuli with T time points and N features.
        R : array_like, shape (T, M)
            Responses with T time points and M separate responses.
        α : float or array_like, shape (M,)
            Regularization parameter. Can be given as a single value (which is applied to
            all M responses) or separate values for each response.

        Returns
        -------
        wˆ : array_like, shape (N, M)
            Linear regression weights.
        """

        # V* = V′
        # UΣV* = S
        U, Σ, V = np.linalg.svd(S, full_matrices=False)

        # U′R
        UR = np.dot(U.T, R)

        # Expand alpha to a collection if it's just a single value
        α = np.ones(R.shape[0]) * α

        # Compute weights for each alpha
        α_ = np.unique(α)
        wˆ = np.zeros((Σ.shape[0], R.shape[0]))

        for a in α_:
            selvox = np.nonzero(α == a)[0]

            # awt = V′ ___Σ___ U′R
            #          Σ² + α²
            fra = np.diag(Σ / (Σ ** 2 + a ** 2))
            URa = UR[:, selvox]

            awt = V.T.dot(fra).dot(URa)
            wˆ[:, selvox] = awt

        return wˆ
