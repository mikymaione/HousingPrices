# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from dataTypes import ElaborationResult


class BaseRidgeRegression(BaseEstimator):
    coef_: numpy.ndarray
    intercetta: numpy.ndarray
    alpha: float = 1.0

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def executeAll(self, S: numpy.ndarray, y: numpy.ndarray, x_test: numpy.ndarray, y_test: numpy.ndarray) -> ElaborationResult:
        self.fit(S, y)
        R = ElaborationResult(self.coef_, self.predict(x_test))

        R.mape = mean_squared_error(y_test, R.y_predict)
        R.r2 = r2_score(y_test, R.y_predict)

        return R

    def score(self, x_test: numpy.ndarray, y_test: numpy.ndarray) -> numpy.float64:
        y_predict = self.predict(x_test)

        return r2_score(y_test, y_predict)

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        y = y.reshape(-1, 1)

        # Compute the weighted arithmetic mean along the specified axis.
        S_wam = numpy.average(X, axis=0)
        y_wam = numpy.average(y, axis=0)

        S = X - S_wam
        y = y - y_wam

        # Normalization is the process of scaling individual samples to have unit norm.
        # This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
        # This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
        # The function normalize provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms.
        S, norm_L2 = preprocessing.normalize(X=S, norm="l2", axis=0, copy=False, return_norm=True)

        self.coef_ = self.calculateWeights(S, y) / norm_L2
        self.intercetta = y_wam - S_wam.dot(self.coef_.T)

        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        y_predict = numpy.dot(X, self.coef_.T) + self.intercetta

        return y_predict

    # abstract
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("Please Implement this method")