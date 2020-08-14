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

import matplotlib.pyplot as plt

from utility.dataSet import DataSet
from utility.dataUtility import DataUtility
from linearRegression.svd import SVD
from linearRegression.lsqr import LSQR
from linearRegression.cholesky import Cholesky


def printPredict(title: str, ɑ: float, error: float) -> None:
    print(f'{title}\t\t\tɑ = {ɑ:.15f}\t\t\t\tMAPE: {error:.2f}%')


def doPrediction(data: DataSet, normalize: bool):
    errors_cholesky = []
    errors_svd = []
    errors_lsqr = []
    alphas = []

    title = f"Linear regression using normalization: {normalize}"
    print(title)

    for ɑ in [0, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.25, 0.26, 0.27, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
              0.9, 1, 1.1, 2, 5, 15]:
        # apprendi pesi tramite Ridge Regression
        cholesky_ = Cholesky()
        svd_ = SVD()
        lsqr_ = LSQR()

        cholesky_.elaborate(S=data.x_train, y=data.y_train, ɑ=ɑ, normalize=normalize)
        svd_.elaborate(S=data.x_train, y=data.y_train, ɑ=ɑ, normalize=normalize)
        lsqr_.elaborate(S=data.x_train, y=data.y_train, ɑ=ɑ, normalize=normalize)

        y_cholesky = cholesky_.predict(data.x_test)
        y_svd = svd_.predict(data.x_test)
        y_lsqr = lsqr_.predict(data.x_test)

        error_svd = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=y_svd)
        error_lsqr = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=y_lsqr)
        error_cholesky = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=y_cholesky)

        errors_cholesky.append(error_cholesky)
        errors_svd.append(error_svd)
        errors_lsqr.append(error_lsqr)
        alphas.append(ɑ)

        printPredict("Cholesky", ɑ, error_cholesky)
        printPredict("SVD", ɑ, error_svd)
        printPredict("LSQR", ɑ, error_lsqr)

    # plt.style.use('grayscale')
    plt.title(title)
    plt.xlabel("Alpha")
    plt.ylabel("MAPE")

    plt.plot(alphas, errors_cholesky, label="Cholesky")
    plt.plot(alphas, errors_svd, label="SVD")
    plt.plot(alphas, errors_lsqr, label="LSQR")

    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    print("HousingPrices Project")
    print("Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione")
    print("")
    print("Elaborating...")

    # carica i dati
    data = DataUtility.load_data(csv_file="cal-housing.csv")

    doPrediction(data=data, normalize=True)
    doPrediction(data=data, normalize=False)
