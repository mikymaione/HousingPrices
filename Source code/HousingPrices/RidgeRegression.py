# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy


class RidgeRegression:
    """
    ||w - Xw||^2_2 + alpha * ||w||2_2
    """

    def __init__(self, alpha: float, reg_strength: float, max_iter: int) -> None:
        self.alpha = alpha
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self.w: numpy.array
        self.S: numpy.array
        self.y_train: numpy.array
        self.m: int
        self.num_feats: int

    def fit(self, x_train: numpy.ndarray, y_train: numpy.ndarray) -> None:
        self.m = x_train.shape[0]
        self.num_feats = x_train.shape[1]

        self.S = numpy.append(numpy.ones(self.m).reshape(-1, 1), x_train, axis=1)

        self.y_train = y_train
        self.w = numpy.zeros(self.num_feats + 1)  # +1 for the intercept

        for _ in range(self.max_iter):
            self.update_step()

    def update_step(self) -> None:
        """
        theta_k = theta_k - (learn_rate * J_theta_k)
        J_theta_k = (2 / train_size) * ( ((w_hat - y_real) * x_k) + (alpha * theta_k^2))
        """
        w_hat = (self.w * self.S).sum(axis=1)
        e = (w_hat - self.y_train).reshape(-1, 1)

        j_theta = (2 / self.m) * ((e * self.S).sum(axis=0) + (self.reg_strength * self.w))
        step = self.alpha * j_theta

        self.w -= step.reshape(-1)

    def predict(self, x_test: numpy.ndarray) -> numpy.ndarray:
        test_size = x_test.shape[0]
        x_test = numpy.append(numpy.ones(test_size).reshape(-1, 1), x_test, axis=1)

        return (self.w * x_test).sum(axis=1)
