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

from tabulate import tabulate

from Utility.dataSet import DataSet, DataElaboration
from Utility.dataUtility import DataUtility
from LinearRegression.RidgeRegression.svd import SVD
from LinearRegression.RidgeRegression.lsqr import LSQR
from LinearRegression.RidgeRegression.cholesky import Cholesky


def doPrediction(data: DataSet, normalize: bool, table) -> DataElaboration:
    R = DataElaboration([], [], [], [], [], [], [])

    for ɑ in [0, 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.25, 0.26, 0.27, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
              1.1, 1.5, 2, 5, 15]:
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

        mape_cholesky = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=y_cholesky)
        mape_svd = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=y_svd)
        mape_lsqr = DataUtility.mean_absolute_percentage_error(y_test=data.y_test, y_predict=y_lsqr)

        r2_cholesky = DataUtility.coefficient_of_determination(y_test=data.y_test, y_predict=y_cholesky)
        r2_svd = DataUtility.coefficient_of_determination(y_test=data.y_test, y_predict=y_svd)
        r2_lsqr = DataUtility.coefficient_of_determination(y_test=data.y_test, y_predict=y_lsqr)

        R.mapes_cholesky.append(mape_cholesky)
        R.mapes_svd.append(mape_svd)
        R.mapes_lsqr.append(mape_lsqr)

        R.r2s_cholesky.append(r2_cholesky)
        R.r2s_svd.append(r2_svd)
        R.r2s_lsqr.append(r2_lsqr)

        R.alphas.append(ɑ)

        table.append([normalize, "Cholesky", ɑ, mape_cholesky, r2_cholesky])
        table.append([normalize, "SVD", ɑ, mape_svd, r2_svd])
        table.append([normalize, "LSQR", ɑ, mape_lsqr, r2_lsqr])

    return R


if __name__ == "__main__":
    print("HousingPrices Project")
    print("Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione")
    print("")
    print("Elaborating...")

    # carica i dati
    data = DataUtility.load_data(csv_file="cal-housing.csv")

    plt.style.use('grayscale')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.canvas.set_window_title('Ridge regression')

    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("MAPE")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("R²")
    ax3.set_xlabel("Alpha")
    ax3.set_ylabel("MAPE")
    ax4.set_xlabel("Alpha")
    ax4.set_ylabel("R²")

    labels = ["Cholesky", "SVD", "LSQR"]
    table = []

    for normalize in [True, False]:
        if normalize:
            asx = ax1
            adx = ax2
        else:
            asx = ax3
            adx = ax4

        P = doPrediction(data=data, normalize=normalize, table=table)

        asx.set_title(f"Normalization: {normalize}")
        asx.plot(P.alphas, P.mapes_cholesky, label=labels[0])
        asx.plot(P.alphas, P.mapes_svd, label=labels[1])
        asx.plot(P.alphas, P.mapes_lsqr, label=labels[2])

        adx.set_title(f"Normalization: {normalize}")
        adx.plot(P.alphas, P.r2s_cholesky, label=labels[0])
        adx.plot(P.alphas, P.r2s_svd, label=labels[1])
        adx.plot(P.alphas, P.r2s_lsqr, label=labels[2])

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    print(tabulate(table, headers=["Normalized", "Algo.", "ɑ", "MAPE", "R²"]))

    plt.tight_layout()
    plt.show()
