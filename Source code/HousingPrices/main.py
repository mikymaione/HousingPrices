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

from svd import SVD
from cholesky import Cholesky
from dataSet import DataSet
from dataUtility import DataUtility


def printPredict(title: str, alpha: float, error: float) -> None:
    print(f'Mean absolute percentage error on test set with alpha {alpha:.15f} using {title}: {error:.2f}%')


if __name__ == "__main__":
    print("HousingPrices Project")
    print("Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione")
    print("")
    print("Elaborating...")

    # carica i dati
    data = DataUtility.load_data(csv_file="cal-housing.csv")

    _cholesky = []
    _svd = []
    _alphas = []

    for ɑ in [0, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.25, 0.26, 0.27, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
              0.9, 1, 1.1, 2, 5, 15]:
        # apprendi pesi tramite Ridge Regression
        cholesky_ = Cholesky()
        svd_ = SVD()

        cholesky_.elaborate(S=data.x_train, y=data.y_train, ɑ=ɑ)
        svd_.elaborate(S=data.x_train, y=data.y_train, ɑ=ɑ)

        e1 = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=cholesky_.predict(data.x_test))
        e2 = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=svd_.predict(data.x_test))

        _alphas.append(ɑ)
        _cholesky.append(e1)
        _svd.append(e2)

        printPredict("Cholesky", ɑ, e1)
        printPredict("SVD", ɑ, data, e2)

    plt.title("Linear regression")
    plt.xlabel("Alpha")
    plt.ylabel("MAPE")

    lGD = plt.plot(_alphas, _cholesky, label="Cholesky")
    lSVD = plt.plot(_alphas, _svd, label="SVD")
    plt.legend(loc="upper left")
    plt.show()
