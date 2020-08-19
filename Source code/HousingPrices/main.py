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

labels = ["Cholesky", "SVD", "LSQR"]


# apprendi pesi tramite Ridge Regression
def doPrediction(R: DataElaboration, ɑ: float, data: DataSet, normalize: bool, tabulateOutput) -> None:
    cholesky_ = Cholesky()
    svd_ = SVD()
    lsqr_ = LSQR()

    mape_cholesky, r2_cholesky = cholesky_.executeAll(S=data.x_train, y=data.y_train, ɑ=ɑ, normalize=normalize,
                                                      x_test=data.x_test, y_test=data.y_test)
    mape_svd, r2_svd = svd_.executeAll(S=data.x_train, y=data.y_train, ɑ=ɑ, normalize=normalize, x_test=data.x_test,
                                       y_test=data.y_test)
    mape_lsqr, r2_lsqr = lsqr_.executeAll(S=data.x_train, y=data.y_train, ɑ=ɑ, normalize=normalize, x_test=data.x_test,
                                          y_test=data.y_test)

    R.mapes_cholesky.append(mape_cholesky)
    R.mapes_svd.append(mape_svd)
    R.mapes_lsqr.append(mape_lsqr)

    R.r2s_cholesky.append(r2_cholesky)
    R.r2s_svd.append(r2_svd)
    R.r2s_lsqr.append(r2_lsqr)

    R.alphas.append(ɑ)

    tabulateOutput.append([normalize, labels[0], ɑ, mape_cholesky, r2_cholesky])
    tabulateOutput.append([normalize, labels[1], ɑ, mape_svd, r2_svd])
    tabulateOutput.append([normalize, labels[2], ɑ, mape_lsqr, r2_lsqr])


def doPredictions(num_set: int, data: DataSet, normalize: bool, tabulateOutput) -> DataElaboration:
    R = DataElaboration(num_set, [], [], [], [], [], [], [], normalize, 100, 100, 100, -1, -1, -1)

    for ɑ in [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.25, 0.26, 0.27, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
              1.1, 1.5, 2, 5, 15]:
        doPrediction(R, ɑ, data, normalize, tabulateOutput)

    R.best_cholesky_alpha, R.min_cholesky_mape = findMinAlpha(R.mapes_cholesky, R.alphas)
    R.best_svd_alpha, R.min_svd_mape = findMinAlpha(R.mapes_svd, R.alphas)
    R.best_lsqr_alpha, R.min_lsqr_mape = findMinAlpha(R.mapes_lsqr, R.alphas)

    return R


def findMinAlpha(lista, alphas):
    best_alpha = alphas[0]
    min_mape = lista[0]

    for i in range(len(lista)):
        if lista[i] < min_mape:
            min_mape = lista[i]
            best_alpha = alphas[i]

    return best_alpha, min_mape


def findMinPredictions(predictions):
    P0 = predictions[0]

    for P in predictions:
        P0 = P
        break

    cholesky_P = P0
    svd_P = P0
    lsqr_P = P0

    cholesky_mape = P0.min_cholesky_mape
    svd_mape = P0.min_svd_mape
    lsqr_mape = P0.min_lsqr_mape

    for P in predictions:
        if P.min_cholesky_mape < cholesky_mape:
            cholesky_mape = P.min_cholesky_mape
            cholesky_P = P

    for P in predictions:
        if P.min_svd_mape < svd_mape:
            svd_mape = P.min_svd_mape
            svd_P = P

    for P in predictions:
        if P.min_lsqr_mape < lsqr_mape:
            lsqr_mape = P.min_lsqr_mape
            lsqr_P = P

    return cholesky_P, svd_P, lsqr_P


def plot_DataElaboration(titolo: str, P: DataElaboration) -> None:
    # plt.style.use('grayscale')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Normalization: {P.normalized}")
    fig.canvas.set_window_title(titolo)

    ax1.set_title("MAPE")
    ax2.set_title("R²")

    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("MAPE")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("R²")

    ax1.plot(P.alphas, P.mapes_cholesky, label=labels[0])
    ax1.plot(P.alphas, P.mapes_svd, label=labels[1])
    ax1.plot(P.alphas, P.mapes_lsqr, label=labels[2])

    ax2.plot(P.alphas, P.r2s_cholesky, label=labels[0])
    ax2.plot(P.alphas, P.r2s_svd, label=labels[1])
    ax2.plot(P.alphas, P.r2s_lsqr, label=labels[2])

    ax1.legend()
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    print("HousingPrices Project")
    print("Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione")
    print("")
    print("Elaborating...")

    column_to_predict = 'median_house_value'
    categories_columns = ['ocean_proximity']
    numerics_columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
                        "households", "median_income"]
    # carica i dati
    cv_datas = DataUtility.load_data("cal-housing.csv", True, column_to_predict, categories_columns, numerics_columns)
    datas = DataUtility.load_data("cal-housing.csv", False, column_to_predict, categories_columns, numerics_columns)

    tabulateOutput = []

    for normalize in [True, False]:
        predictions = []

        for i in range(len(cv_datas)):
            data = cv_datas[i]
            P = doPredictions(i, data=data, normalize=normalize, tabulateOutput=tabulateOutput)
            predictions.append(P)

        cholesky_P, svd_P, lsqr_P = findMinPredictions(predictions)

        data = datas[0]
        cholesky_ = Cholesky()
        svd_ = SVD()
        lsqr_ = LSQR()

        mape_cholesky, r2_cholesky = cholesky_.executeAll(S=data.x_train, y=data.y_train,
                                                          ɑ=cholesky_P.best_cholesky_alpha, normalize=normalize,
                                                          x_test=data.x_test, y_test=data.y_test, showPlot=True,
                                                          plotTitle=f"Cholesky, Normalization: {normalize}")
        mape_svd, r2_svd = svd_.executeAll(S=data.x_train, y=data.y_train, ɑ=svd_P.best_svd_alpha, normalize=normalize,
                                           x_test=data.x_test, y_test=data.y_test, showPlot=True,
                                           plotTitle=f"SVD, Normalization: {normalize}")
        mape_lsqr, r2_lsqr = lsqr_.executeAll(S=data.x_train, y=data.y_train, ɑ=lsqr_P.best_lsqr_alpha,
                                              normalize=normalize, x_test=data.x_test, y_test=data.y_test,
                                              showPlot=True, plotTitle=f"LSQR, Normalization: {normalize}")

    print(tabulate(tabulateOutput, headers=["Normalized", "Algo.", "ɑ", "MAPE", "R²"]))
