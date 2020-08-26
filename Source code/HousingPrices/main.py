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

from typing import List
from tabulate import tabulate

from Utility.plotting import Plotting
from Utility.dataTypes import DataSet, DataElaboration
from Utility.dataManager import DataManager
from Utility.dataFunctions import DataFunctions

from LinearRegression.RidgeRegression.ridge_sklearn import Ridge_SKLearn
from LinearRegression.RidgeRegression.svd import SVD
from LinearRegression.RidgeRegression.lsqr import LSQR
from LinearRegression.RidgeRegression.cholesky import Cholesky

labels = ["Cholesky", "SVD", "LSQR", "SKLearn"]
alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.25, 0.26, 0.27, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.5, 2, 5, 15, 17]


# apprendi pesi tramite Ridge Regression
def doPrediction(R: DataElaboration, ɑ: float, data: DataSet, tabulateOutput) -> None:
    cholesky_ = Cholesky()
    svd_ = SVD()
    lsqr_ = LSQR()
    ridge_sklearn_ = Ridge_SKLearn()

    R_ridge_sklearn = ridge_sklearn_.executeAll(S=data.x_train, y=data.y_train, ɑ=ɑ, x_test=data.x_test, y_test=data.y_test)
    R_cholesky = cholesky_.executeAll(S=data.x_train, y=data.y_train, ɑ=ɑ, x_test=data.x_test, y_test=data.y_test)
    R_svd = svd_.executeAll(S=data.x_train, y=data.y_train, ɑ=ɑ, x_test=data.x_test, y_test=data.y_test)
    R_lsqr = lsqr_.executeAll(S=data.x_train, y=data.y_train, ɑ=ɑ, x_test=data.x_test, y_test=data.y_test)

    R.w_ridge_sklearn.append(R_ridge_sklearn.w)
    R.w_cholesky.append(R_cholesky.w)
    R.w_svd.append(R_svd.w)
    R.w_lsqr.append(R_lsqr.w)

    R.mapes_ridge_sklearn.append(R_ridge_sklearn.mape)
    R.mapes_cholesky.append(R_cholesky.mape)
    R.mapes_svd.append(R_svd.mape)
    R.mapes_lsqr.append(R_lsqr.mape)

    R.r2s_ridge_sklearn.append(R_ridge_sklearn.r2)
    R.r2s_cholesky.append(R_cholesky.r2)
    R.r2s_svd.append(R_svd.r2)
    R.r2s_lsqr.append(R_lsqr.r2)

    R.alphas.append(ɑ)

    tabulateOutput.append([labels[0], ɑ, R_cholesky.mape, R_cholesky.r2])
    tabulateOutput.append([labels[1], ɑ, R_svd.mape, R_svd.r2])
    tabulateOutput.append([labels[2], ɑ, R_lsqr.mape, R_lsqr.r2])
    tabulateOutput.append([labels[3], ɑ, R_ridge_sklearn.mape, R_ridge_sklearn.r2])


def doPredictions(num_set: int, data: DataSet, tabulateOutput) -> DataElaboration:
    R = DataElaboration(num_set)

    for ɑ in alphas:
        doPrediction(R, ɑ, data, tabulateOutput)

    R.best_cholesky_alpha, R.min_cholesky_mape = DataFunctions.findMinAlpha(R.mapes_cholesky, R.alphas)
    R.best_svd_alpha, R.min_svd_mape = DataFunctions.findMinAlpha(R.mapes_svd, R.alphas)
    R.best_lsqr_alpha, R.min_lsqr_mape = DataFunctions.findMinAlpha(R.mapes_lsqr, R.alphas)
    R.best_ridge_sklearn_alpha, R.min_ridge_sklearn_mape = DataFunctions.findMinAlpha(R.mapes_ridge_sklearn, R.alphas)

    # Plotting.plot_DataElaboration(f"Ridge regression, Normalization: {normalize}", labels, R)

    return R


def executeCrossValidation() -> None:
    # carica i dati
    cv_datas = DataManager.load_data("cal-housing.csv", True)
    datas = DataManager.load_data("cal-housing.csv", False)

    tabulateOutput = []

    for normalize in [True, False]:
        predictions: List[DataElaboration] = []

        for i in range(len(cv_datas)):
            data = cv_datas[i]
            P = doPredictions(i, data=data, tabulateOutput=tabulateOutput)
            predictions.append(P)

        minPredictions = DataFunctions.findMinPredictions(predictions)

        executeOnMinPrediction(datas[0], normalize, minPredictions)

    print(tabulate(tabulateOutput, headers=["Normalized", "Algo.", "ɑ", "MAPE", "R²"]))


def executeOnRangeOfAlpha() -> None:
    # carica i dati
    datas, X = DataManager.load_data("cal-housing.csv", False)
    data = datas[0]

    tabulateOutput = []

    P = doPredictions(0, data, tabulateOutput)
    # DataManager.doPCA(X, P.w_ridge_sklearn)

    Plotting.plot_DataElaboration("Ridge regression", labels, P)

    minPredictions = DataFunctions.findMinPredictions([P])
    executeOnMinPrediction(data, minPredictions)

    print(tabulate(tabulateOutput, headers=["Algo.", "ɑ", "MAPE", "R²"]))


def executeOnMinPrediction(data: DataSet, minPredictions: DataElaboration):
    cholesky_ = Cholesky()
    svd_ = SVD()
    lsqr_ = LSQR()
    ridge_sklearn_ = Ridge_SKLearn()

    R_cholesky = cholesky_.executeAll(S=data.x_train, y=data.y_train, ɑ=minPredictions.best_cholesky_alpha, x_test=data.x_test, y_test=data.y_test)
    R_svd = svd_.executeAll(S=data.x_train, y=data.y_train, ɑ=minPredictions.best_svd_alpha, x_test=data.x_test, y_test=data.y_test)
    R_lsqr = lsqr_.executeAll(S=data.x_train, y=data.y_train, ɑ=minPredictions.best_lsqr_alpha, x_test=data.x_test, y_test=data.y_test)
    R_ridge_sklearn = ridge_sklearn_.executeAll(S=data.x_train, y=data.y_train, ɑ=minPredictions.best_lsqr_alpha, x_test=data.x_test, y_test=data.y_test)

    Plotting.scatterPlot(f"Cholesky - ɑ: {minPredictions.best_cholesky_alpha}", y_predict=R_cholesky.y_predict, y_test=data.y_test)
    Plotting.scatterPlot(f"SVD - ɑ: {minPredictions.best_svd_alpha}", y_predict=R_svd.y_predict, y_test=data.y_test)
    Plotting.scatterPlot(f"LSQR - ɑ: {minPredictions.best_lsqr_alpha}", y_predict=R_lsqr.y_predict, y_test=data.y_test)
    Plotting.scatterPlot(f"SKLearn - ɑ: {minPredictions.best_ridge_sklearn_alpha}", y_predict=R_ridge_sklearn.y_predict, y_test=data.y_test)


if __name__ == "__main__":
    print("HousingPrices Project")
    print("Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione")
    print("")
    print("Elaborating...")

    #executeOnRangeOfAlpha()
    executeCrossValidation()
