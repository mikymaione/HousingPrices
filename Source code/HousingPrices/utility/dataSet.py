# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from dataclasses import dataclass


@dataclass
class DataElaboration:
    num_set: int

    alphas: list

    mapes_cholesky: list
    mapes_svd: list
    mapes_lsqr: list

    r2s_cholesky: list
    r2s_svd: list
    r2s_lsqr: list

    normalized: bool

    min_cholesky_mape: numpy.float64
    min_svd_mape: numpy.float64
    min_lsqr_mape: numpy.float64

    best_cholesky_alpha: float
    best_svd_alpha: float
    best_lsqr_alpha: float


@dataclass
class DataSet:
    x_train: numpy.ndarray
    x_test: numpy.ndarray
    y_train: numpy.ndarray
    y_test: numpy.ndarray
