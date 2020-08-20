# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from typing import List, Tuple

from Utility.dataTypes import DataElaboration


class DataFunctions:

    @staticmethod
    def findMinAlpha(mapes: List[numpy.float64], alphas: List[float]) -> Tuple[float, numpy.float64]:
        best_alpha = alphas[0]
        min_mape = mapes[0]

        for i in range(len(mapes)):
            if mapes[i] < min_mape:
                min_mape = mapes[i]
                best_alpha = alphas[i]

        return best_alpha, min_mape

    @staticmethod
    def findMinPredictions(predictions: List[DataElaboration]) -> DataElaboration:
        P0 = predictions[0]

        for P in predictions:
            if P.min_cholesky_mape < P0.min_cholesky_mape:
                P0.min_cholesky_mape = P.min_cholesky_mape

            if P.min_svd_mape < P0.min_svd_mape:
                P0.min_svd_mape = P.min_svd_mape

            if P.min_lsqr_mape < P0.min_lsqr_mape:
                P0.min_lsqr_mape = P.min_lsqr_mape

            if P.min_ridge_sklearn_mape < P0.min_ridge_sklearn_mape:
                P0.min_ridge_sklearn_mape = P.min_ridge_sklearn_mape

        return P0
