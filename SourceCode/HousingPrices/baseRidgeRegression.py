# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy
import pandas
from joblib import delayed, Parallel

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split, learning_curve, cross_validate, ParameterGrid, ParameterSampler
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from dataTypes import ElaborationResult


class BaseRidgeRegression(BaseEstimator):
    nested_cross_validation_trials = 10

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.R2 = []
        self.MSE = []

    def nestedCrossValidation(self, X: pandas.DataFrame, y: pandas.Series):
        EX_CV = KFold(n_splits=self.nested_cross_validation_trials, shuffle=True, random_state=1986)
        IN_CV = KFold(n_splits=5, shuffle=True)

        outer_scores = []
        variance = []
        best_inner_list = []
        best_inner_alpha = 0

        for (i, (EX_train_idx, EX_test_idx)) in enumerate(EX_CV.split(X, y)):
            EX_x_train, EX_x_test = X.iloc[EX_train_idx], X.iloc[EX_test_idx]
            EX_y_train, EX_y_test = y.iloc[EX_train_idx], y.iloc[EX_test_idx]

            for (j, (IN_train_idx, IN_test_idx)) in enumerate(IN_CV.split(EX_x_train, EX_y_train)):
                def _fit(x_train: pandas.DataFrame, x_test: pandas.DataFrame, y_train: pandas.Series, y_test: pandas.Series, ɑ: float):
                    self.alpha = ɑ
                    self.fit(x_train, y_train)
                    inner_grid_score = self.score(x_test, y_test)
                    return inner_grid_score, ɑ

                IN_x_train, IN_x_test = EX_x_train.iloc[IN_train_idx], EX_x_train.iloc[IN_test_idx]
                IN_y_train, IN_y_test = EX_y_train.iloc[IN_train_idx], EX_y_train.iloc[IN_test_idx]

                IN_result = Parallel(n_jobs=-1)(delayed(_fit)(IN_x_train, IN_x_test, IN_y_train, IN_y_test, ɑ)
                                                for ɑ in self.alphas)

                IN_result = numpy.array(IN_result)
                best_idx = numpy.argmax(IN_result[:, 0])
                best_inner_alpha = IN_result[best_idx][1]
                best_inner_list.append(IN_result[best_idx])

            self.alpha = best_inner_alpha
            self.fit(EX_x_train, EX_y_train)

            score, pred = self.score_and_prediction(EX_x_test, EX_y_test)
            outer_scores.append(score)
            variance.append(numpy.var(pred))

        best_inner_list = numpy.array(best_inner_list)
        best_idx = numpy.argmax(best_inner_list[:, 0])
        best_alpha = best_inner_list[best_idx][1]

        self.best_alpha = best_alpha

        return numpy.array(variance), numpy.array(outer_scores), best_alpha

    def nestedCrossValidationKFold(self, X, y) -> None:
        p_grid = {"alpha": self.alphas}

        self.non_nested_scores = numpy.zeros(self.nested_cross_validation_trials)
        self.nested_scores = numpy.zeros(self.nested_cross_validation_trials)

        for i in range(self.nested_cross_validation_trials):
            inner_cv = KFold(n_splits=5, shuffle=True)
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=1986)

            clf = GridSearchCV(estimator=self, param_grid=p_grid, cv=inner_cv, n_jobs=-1)
            clf.fit(X, y)

            self.non_nested_scores[i] = clf.best_score_

            self.nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, n_jobs=-1)
            self.nested_scores[i] = self.nested_score.mean()

        self.score_difference = self.non_nested_scores - self.nested_scores

    def crossValidationKFold(self, X: pandas.DataFrame, y: pandas.Series) -> None:
        self.R2.clear()
        self.MSE.clear()

        kf = KFold(n_splits=5, shuffle=True, random_state=1986)

        for ɑ in self.alphas:
            k_scores = []
            k_mses = []

            self.alpha = ɑ

            for train_index, test_index in kf.split(X):
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                self.fit(x_train, y_train)

                y_pred = self.predict(x_test)

                score = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                k_scores.append(score)
                k_mses.append(mse)

            self.R2.append(numpy.mean(k_scores))
            self.MSE.append(numpy.mean(k_mses))

    def printBestScores(self) -> None:
        print(self.algo + ':')
        print(f'-best ɑ: {self.best_alpha}')
        print(f'-best MSE: {self.best_MSE}')
        print(f'-best R²: {self.best_R2}')

    def calculateScoring(self, alphas, x_train: pandas.DataFrame, y_train: pandas.Series, x_test: pandas.DataFrame, y_test: pandas.Series) -> None:
        self.MSE.clear()
        self.R2.clear()
        self.alphas = alphas

        for ɑ in self.alphas:
            self.alpha = ɑ
            self.fit(x_train, y_train)
            y_predict = self.predict(x_test)

            self.MSE.append(mean_squared_error(y_test, y_predict))
            self.R2.append(r2_score(y_test, y_predict))

        idx_best_r2 = numpy.argmax(self.R2)
        idx_best_mse = numpy.argmin(self.MSE)

        self.best_MSE = self.MSE[idx_best_mse]
        self.best_R2 = self.R2[idx_best_r2]

        self.best_alpha = self.alphas[idx_best_r2]
        self.alpha = self.best_alpha

    def trainBySize(self, sizes, X, y) -> numpy.ndarray:
        coef_list = []

        for s in sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=s, shuffle=True, random_state=1986)
            self.fit(X_train, y_train)
            coef_list.append(self.coef_)

        return numpy.array(coef_list)

    def fitPCA(self, n_components: int, X: numpy.ndarray):
        pca = PCA(n_components=n_components)
        pca.fit(X)
        coef_pca = pca.transform(X)
        self.pca_singular_values = pca.singular_values_

        return coef_pca

    def learningCurvePCA(self, sizes, n_components: int, X: numpy.ndarray, y: numpy.ndarray, scoring):
        X_pca = self.fitPCA(n_components, X)
        return learning_curve(self, X_pca, y, train_sizes=sizes, cv=5, scoring=scoring, n_jobs=-1)

    def fitPCABySize(self, sizes, X, y, n_components: int):
        coef_matrix = self.trainBySize(sizes, X, y)
        coef_pca = self.fitPCA(n_components, coef_matrix)

        return coef_pca

    def executeAll(self, S: pandas.DataFrame, y: pandas.Series, x_test: pandas.DataFrame, y_test: pandas.Series) -> ElaborationResult:
        S_ = S.to_numpy()
        y_ = y.to_numpy()
        x_test_ = x_test.to_numpy()
        y_test_ = y_test.to_numpy()

        self.fit(S_, y_)
        R = ElaborationResult(self.coef_, self.predict(x_test_))

        R.mape = mean_squared_error(y_test_, R.y_predict)
        R.r2 = r2_score(y_test_, R.y_predict)

        return R

    def score_and_prediction(self, x_test: pandas.DataFrame, y_test: pandas.Series):
        y_predict = self.predict(x_test)
        score = r2_score(y_test, y_predict)

        return score, y_predict

    def score(self, x_test: pandas.DataFrame, y_test: pandas.Series) -> numpy.float64:
        score, y_predict = self.score_and_prediction(x_test, y_test)

        return score

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
        self.intercetta = y_wam - S_wam @ self.coef_.T

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

        y_predict = X @ self.coef_.T + self.intercetta

        return y_predict

    # abstract
    def calculateWeights(self, S: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("Please Implement this method")
