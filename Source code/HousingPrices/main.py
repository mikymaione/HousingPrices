# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# HousingPrices Project
# Implement from scratch the ridge regression algorithm for regression with square loss.
# Apply the algorithm to the prediction of the label medianHouseValue in this dataset.
# Study the dependence of the cross-validated risk estimate on the parameter alpha of ridge regression.
# Try using PCA to improve the risk estimate.
# Optionally, use nested cross-validated risk estimates to remove the need of choosing the parameter.

import numpy
import matplotlib.pyplot as plt

from dataUtility import DataUtility
from ridgeRegression import RidgeRegression
from dataSet import DataSet


def printPredict(title: str, alpha: float, data: DataSet, w: numpy.ndarray) -> numpy.float64:
    # esegui una predizione
    y_predict = RidgeRegression.predict(x_test=data.x_test, w=w)

    # calcola errore
    error = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=y_predict)

    print(f'Mean absolute percentage error on test set with alpha {alpha:.1f} using {title}: {error:.2f}%')

    return error


if __name__ == "__main__":
    print("HousingPrices Project")
    print("Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione")
    print("")
    print("Elaborating...")

    # carica i dati
    data = DataUtility.load_data(csv_file="cal-housing.csv")

    gradientDescent = []
    svd = []
    alphas = []

    for alpha in numpy.arange(0.1, 1.1, 0.1):
        # apprendi pesi tramite Ridge Regression
        w1 = RidgeRegression.gradient_descent(S=data.x_train, y=data.y_train, alpha=alpha)
        w2 = RidgeRegression.svd(S=data.x_train, y=data.y_train, alpha=alpha)

        e1 = printPredict("Gradient Descent", alpha, data, w1)
        e2 = printPredict("SVD", alpha, data, w2)

        alphas.append(alpha)
        gradientDescent.append(e1)
        svd.append(e2)

    plt.plot(alphas, gradientDescent)
    plt.plot(alphas, svd)
    plt.show()
