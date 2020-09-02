# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn import decomposition
from typing import List, Tuple

from dataTypes import DataSet

shuffleDataSet = False
column_to_predict = 'median_house_value'
categories_columns = ['ocean_proximity']
numerics_columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]


class DataManager:

    # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    @staticmethod
    def load_data(csv_file: str, use_cross_validation: bool, usePCA: bool = False, showInfo: bool = False) -> Tuple[List[DataSet], pandas.DataFrame, pandas.Series]:
        # shuffled datasets
        datasets: List[DataSet] = []

        # leggi tutto il file
        data_frame = pandas.read_csv(filepath_or_buffer=csv_file)
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)

        if showInfo:
            print(data_frame.info())
            print(data_frame.describe())
            print(data_frame.head())

            seaborn.distplot(data_frame[column_to_predict])

        # metti media in celle vuote
        for c in data_frame.columns:
            if data_frame[c].hasnans:
                m = data_frame[c].mean()
                data_frame[c].fillna(value=m, inplace=True)

        labelencoder = LabelEncoder()

        for c in categories_columns:
            data_frame[c + '_cat'] = labelencoder.fit_transform(data_frame[c])

        data_frame.drop(columns=categories_columns, inplace=True)

        outliers = data_frame[data_frame[column_to_predict] == 500001].index
        data_frame.drop(outliers, inplace=True)

        if showInfo:
            corr = data_frame.corr()
            seaborn.heatmap(corr, square=True, annot=True)
            plt.title('Correlation between features')

            j = corr[((corr > 0.75) | (corr < -0.75)) & (corr != 1.0)].dropna(axis='index', how='all')
            print(j)

        columns_to_use = list(data_frame.columns)

        for u in ['households', 'total_bedrooms', column_to_predict]:
            columns_to_use.remove(u)
            if numerics_columns.count(u) > 0:
                numerics_columns.remove(u)

        X = data_frame.drop(columns=['households', 'total_bedrooms', column_to_predict])
        y = data_frame[column_to_predict]

        column_to_predict_idx = data_frame.columns.get_loc(column_to_predict)
        cols = list(range(0, data_frame.shape[1]))
        cols.remove(column_to_predict_idx)

        scaler = MinMaxScaler()
        scaler.fit(data_frame)
        data_frame = scaler.transform(data_frame)

        X = data_frame[:, cols]
        y = data_frame[:, column_to_predict_idx]

        if usePCA:
            X = DataManager.getPCA(X)

        # dividi in X e y, sia di train che test
        if use_cross_validation:
            # K-Folds cross-validator
            # Provides train/test indices to split data in train/test sets.
            # Split dataset into k consecutive folds (without shuffling by default).
            # Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
            kf = KFold(n_splits=6, shuffle=shuffleDataSet)
            X = X.to_numpy()

            for train_index, test_index in kf.split(X):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                datasets.append(DataManager.toDataSet(x_train, x_test, y_train, y_test, not usePCA))
        else:
            # Split arrays or matrices into random train and test subsets
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=shuffleDataSet)

            datasets.append(DataManager.toDataSet(x_train, x_test, y_train, y_test, not usePCA))

        return datasets, X, y

    @staticmethod
    def getPCA(X: pandas.DataFrame) -> numpy.array:
        pca = decomposition.PCA(n_components=2)
        pca.fit(X)

        X_pca = pca.transform(X)

        return X_pca

    @staticmethod
    def doPCA(X: pandas.DataFrame, coef_list: List[numpy.array]) -> None:
        coef_matrix = numpy.array(coef_list)

        pca = decomposition.PCA(n_components=2)
        pca.fit(coef_matrix)
        coef_pca = pca.transform(coef_matrix)

        plt.scatter(coef_pca[:, 0], coef_pca[:, 1])
        plt.show()

        pca = decomposition.PCA(n_components=2)
        pca.fit(X)

        plt.title('PCA')
        plt.plot(pca.singular_values_, label='Singular values')
        plt.legend()
        plt.show()

    @staticmethod
    def toDataSet(x_train, x_test, y_train: pandas.Series, y_test: pandas.Series, standardize: bool) -> DataSet:
        # if standardize:
        # Standardize features by removing the mean and scaling to unit variance
        # The standard score of a sample X is calculated as:
        # z = (X - μ) / σ²
        # ss = StandardScaler()
        # ss.fit(X=x_train[numerics_columns])
        # x_train = ss.transform(X=x_train[numerics_columns])
        #
        # # usa (X-μ / √σ²) sul test
        # centered_df = x_test[numerics_columns] - ss.mean_
        # x_test = centered_df / (ss.var_ ** 0.5)
        # x_test = x_test.to_numpy()

        return DataSet(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train.to_numpy(),
            y_test=y_test.to_numpy()
        )

    @staticmethod
    def __categories_to_columns(data_frame: pandas.DataFrame, categories_columns: List[str]) -> pandas.DataFrame:
        columns = pandas.DataFrame()

        for c in categories_columns:
            column = pandas.get_dummies(data=data_frame[c], prefix=c + '_')
            columns = pandas.concat((columns, column), axis=1)

        return columns
