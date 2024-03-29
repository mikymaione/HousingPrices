# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy
import pandas

from typing import List

from dataclasses import dataclass, field


@dataclass
class ElaborationResult:
    w: numpy.ndarray
    y_predict: numpy.ndarray
    mape: numpy.float64 = 0
    r2: numpy.float64 = 0


@dataclass
class DataElaboration:
    num_set: int

    alphas: List[float] = field(default_factory=list)

    w_cholesky: List[numpy.ndarray] = field(default_factory=list)
    w_svd: List[numpy.ndarray] = field(default_factory=list)
    w_lsqr: List[numpy.ndarray] = field(default_factory=list)

    mapes_cholesky: List[numpy.float64] = field(default_factory=list)
    mapes_svd: List[numpy.float64] = field(default_factory=list)
    mapes_lsqr: List[numpy.float64] = field(default_factory=list)

    r2s_cholesky: List[numpy.float64] = field(default_factory=list)
    r2s_svd: List[numpy.float64] = field(default_factory=list)
    r2s_lsqr: List[numpy.float64] = field(default_factory=list)

    min_cholesky_mape: numpy.float64 = 100
    min_svd_mape: numpy.float64 = 100
    min_lsqr_mape: numpy.float64 = 100

    best_cholesky_alpha: float = -1
    best_svd_alpha: float = -1
    best_lsqr_alpha: float = -1


@dataclass
class DataSet:
    x_train: pandas.DataFrame
    x_test: pandas.DataFrame
    y_train: pandas.Series
    y_test: pandas.Series
