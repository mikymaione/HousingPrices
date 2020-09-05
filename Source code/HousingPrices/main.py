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

from tabulate import tabulate
from typing import List
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

from plotting import Plotting
from dataTypes import DataSet, DataElaboration
from dataManager import DataManager
from dataFunctions import DataFunctions

from svd import SVD
from lsqr import LSQR
from cholesky import Cholesky

labels = ["Cholesky", "SVD", "LSQR"]
alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.25, 0.26, 0.27, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.5, 2, 5, 15, 17]


# apprendi pesi tramite Ridge Regression
def doPrediction(R: DataElaboration, ɑ: float, data: DataSet, tabulateOutput) -> None:
    cholesky_ = Cholesky(ɑ)
    svd_ = SVD(ɑ)
    lsqr_ = LSQR(ɑ)

    R_cholesky = cholesky_.executeAll(S=data.x_train, y=data.y_train, x_test=data.x_test, y_test=data.y_test)
    R_svd = svd_.executeAll(S=data.x_train, y=data.y_train, x_test=data.x_test, y_test=data.y_test)
    R_lsqr = lsqr_.executeAll(S=data.x_train, y=data.y_train, x_test=data.x_test, y_test=data.y_test)

    R.w_cholesky.append(R_cholesky.w)
    R.w_svd.append(R_svd.w)
    R.w_lsqr.append(R_lsqr.w)

    R.mapes_cholesky.append(R_cholesky.mape)
    R.mapes_svd.append(R_svd.mape)
    R.mapes_lsqr.append(R_lsqr.mape)

    R.r2s_cholesky.append(R_cholesky.r2)
    R.r2s_svd.append(R_svd.r2)
    R.r2s_lsqr.append(R_lsqr.r2)

    R.alphas.append(ɑ)

    tabulateOutput.append([labels[0], ɑ, R_cholesky.mape, R_cholesky.r2])
    tabulateOutput.append([labels[1], ɑ, R_svd.mape, R_svd.r2])
    tabulateOutput.append([labels[2], ɑ, R_lsqr.mape, R_lsqr.r2])


def doPredictions(num_set: int, data: DataSet, tabulateOutput) -> DataElaboration:
    R = DataElaboration(num_set)

    for ɑ in alphas:
        doPrediction(R, ɑ, data, tabulateOutput)

    R.best_cholesky_alpha, R.min_cholesky_mape = DataFunctions.findMinAlpha(R.mapes_cholesky, R.alphas)
    R.best_svd_alpha, R.min_svd_mape = DataFunctions.findMinAlpha(R.mapes_svd, R.alphas)
    R.best_lsqr_alpha, R.min_lsqr_mape = DataFunctions.findMinAlpha(R.mapes_lsqr, R.alphas)

    # Plotting.plot_DataElaboration(f"Ridge regression, Normalization: {normalize}", labels, R)

    return R


def executeCrossValidation() -> None:
    # carica i dati
    cv_datas, cv_X, cv_y = DataManager.load_data("cal-housing.csv", True)
    datas, X, y = DataManager.load_data("cal-housing.csv", False)

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
    datas, X, y = DataManager.load_data("cal-housing.csv", False)
    data = datas[0]

    tabulateOutput = []

    P = doPredictions(0, data, tabulateOutput)
    # DataManager.doPCA(X, P.w_ridge_sklearn)

    Plotting.plot_DataElaboration("Ridge regression", labels, P)

    minPredictions = DataFunctions.findMinPredictions([P])
    executeOnMinPrediction(data, minPredictions)

    print(tabulate(tabulateOutput, headers=["Algo.", "ɑ", "MAPE", "R²"]))


def executeOnMinPrediction(data: DataSet, minPredictions: DataElaboration):
    cholesky_ = Cholesky(minPredictions.best_cholesky_alpha)
    svd_ = SVD(minPredictions.best_svd_alpha)
    lsqr_ = LSQR(minPredictions.best_lsqr_alpha)

    R_cholesky = cholesky_.executeAll(S=data.x_train, y=data.y_train, x_test=data.x_test, y_test=data.y_test)
    R_svd = svd_.executeAll(S=data.x_train, y=data.y_train, x_test=data.x_test, y_test=data.y_test)
    R_lsqr = lsqr_.executeAll(S=data.x_train, y=data.y_train, x_test=data.x_test, y_test=data.y_test)

    Plotting.scatterPlot(f"Cholesky - ɑ: {minPredictions.best_cholesky_alpha}", y_predict=R_cholesky.y_predict, y_test=data.y_test)
    Plotting.scatterPlot(f"SVD - ɑ: {minPredictions.best_svd_alpha}", y_predict=R_svd.y_predict, y_test=data.y_test)
    Plotting.scatterPlot(f"LSQR - ɑ: {minPredictions.best_lsqr_alpha}", y_predict=R_lsqr.y_predict, y_test=data.y_test)


def nestedCrossValidation():
    nested_cross_validation_trials = 10
    p_grid = {"alpha": alphas}
    shuffleDataSet = True

    non_nested_scores = numpy.zeros(nested_cross_validation_trials)
    nested_scores = numpy.zeros(nested_cross_validation_trials)

    datas, X, y = DataManager.load_data("cal-housing.csv", False)

    solver = SVD()

    for i in range(nested_cross_validation_trials):
        # Choose cross-validation techniques for the inner and outer loops, independently of the dataset
        inner_cv = KFold(n_splits=5, shuffle=shuffleDataSet)
        outer_cv = KFold(n_splits=5, shuffle=shuffleDataSet)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=solver, param_grid=p_grid, cv=inner_cv, n_jobs=-1)
        clf.fit(X, y)

        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, n_jobs=-1)
        nested_scores[i] = nested_score.mean()

    score_difference = non_nested_scores - nested_scores
    print("Average difference of {:6f} with std. dev. of {:6f}.".format(score_difference.mean(), score_difference.std()))

    # Plot scores on each trial for nested and non-nested CV
    plt.subplot(211)

    non_nested_scores_line, = plt.plot(non_nested_scores)
    nested_line, = plt.plot(nested_scores)

    plt.ylabel("score")
    plt.grid()
    plt.legend([non_nested_scores_line, nested_line], ["Non-Nested CV", "Nested CV"])
    plt.title("Non-Nested and Nested Cross Validation")

    # Plot bar chart of the difference.
    plt.subplot(212)
    difference_plot = plt.bar(range(nested_cross_validation_trials), score_difference)

    plt.xlabel("Individual Trial #")

    plt.grid()
    plt.legend([difference_plot], ["Non-Nested CV - Nested CV Score"])
    plt.ylabel("score difference")

    plt.show()


if __name__ == "__main__":
    print("HousingPrices Project")
    print("Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione")
    print("")
    print("Elaborating...")

    # executeOnRangeOfAlpha()
    # executeCrossValidation()
    # nestedCrossValidation()

    datas, X, y = DataManager.load_data("cal-housing.csv", False)
    data = datas[0]

    best_cholesky_alpha = 0.00001
    cholesky = Cholesky(best_cholesky_alpha)

    R = cholesky.executeAll(S=data.x_train, y=data.y_train, x_test=data.x_test, y_test=data.y_test)

    Plotting.scatterPlot(f"Cholesky - ɑ: {best_cholesky_alpha}", y_predict=R.y_predict, y_test=data.y_test.to_numpy())
