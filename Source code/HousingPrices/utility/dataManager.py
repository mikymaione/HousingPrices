# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy
import pandas

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List

from Utility.dataTypes import DataSet

shuffleDataSet = True


class DataManager:

    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    @staticmethod
    def coefficient_of_determination(y_test: numpy.ndarray, y_predict: numpy.ndarray) -> numpy.float64:
        correlation_matrix = numpy.corrcoef(y_test, y_predict)

        correlation_xy = correlation_matrix[0, 1]

        r_squared = correlation_xy ** 2

        return r_squared

    # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    @staticmethod
    def mean_absolute_percentage_error(y_test: numpy.ndarray, y_predict: numpy.ndarray) -> numpy.float64:
        abs_errors = numpy.abs((y_test - y_predict) / y_test)

        return numpy.mean(abs_errors) * 100

    # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    @staticmethod
    def load_data(csv_file: str, use_cross_validation: bool, column_to_predict: str, categories_columns: List[str], numerics_columns: List[str]) -> List[DataSet]:

        # shuffled datasets
        datasets: List[DataSet] = []

        # leggi tutto il file
        data_frame = pandas.read_csv(filepath_or_buffer=csv_file)

        # metti media in celle vuote
        for c in data_frame.columns:
            if data_frame[c].hasnans:
                m = data_frame[c].mean()
                data_frame[c].fillna(value=m, inplace=True)

        # genera le colonne per ogni elemento di una colonna categoria
        columns_categories = DataManager.__categories_to_columns(data_frame=data_frame, categories_columns=categories_columns)

        # elimina le colonne categoria
        data_frame.drop(columns=categories_columns, inplace=True)

        # aggiungi le colonne per ogni elemento di una colonna categoria
        data_frame = pandas.concat([data_frame, columns_categories], axis=1)

        columns_to_use = list(data_frame.columns)
        columns_to_use.remove(column_to_predict)

        X = data_frame[columns_to_use].to_numpy()
        y = data_frame[column_to_predict].to_numpy()

        # dividi in X e y, sia di train che test
        if use_cross_validation:
            # K-Folds cross-validator
            # Provides train/test indices to split data in train/test sets.
            # Split dataset into k consecutive folds (without shuffling by default).
            # Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
            kf = KFold(n_splits=6, shuffle=shuffleDataSet)

            for train_index, test_index in kf.split(X):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                datasets.append(DataManager.__to_DataSet(x_train, x_test, y_train, y_test, columns_to_use, numerics_columns))
        else:
            # Split arrays or matrices into random train and test subsets
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=shuffleDataSet)

            datasets.append(DataManager.__to_DataSet(x_train, x_test, y_train, y_test, columns_to_use, numerics_columns))

        return datasets

    @staticmethod
    def __to_DataSet(x_train: numpy.ndarray, x_test: numpy.ndarray, y_train: numpy.ndarray, y_test: numpy.ndarray, columns_to_use: List[str], numerics_columns: List[str]) -> DataSet:
        # aggiunti titoli a colonne
        x_train = pandas.DataFrame(x_train, columns=columns_to_use)
        x_test = pandas.DataFrame(x_test, columns=columns_to_use)

        # Standardize features by removing the mean and scaling to unit variance
        # The standard score of a sample X is calculated as:
        # z = (X - μ) / σ²
        ss = StandardScaler()
        ss.fit(X=x_train[numerics_columns])
        x_train = ss.transform(X=x_train[numerics_columns])

        # usa (X-μ / √σ²) sul test
        centered_df = x_test[numerics_columns] - ss.mean_
        x_test = centered_df / (ss.var_ ** 0.5)

        return DataSet(
            x_train=x_train,
            x_test=x_test.to_numpy(),
            y_train=y_train,
            y_test=y_test
        )

    @staticmethod
    def __categories_to_columns(data_frame: pandas.DataFrame, categories_columns: List[str]) -> pandas.DataFrame:
        columns = pandas.DataFrame()

        for c in categories_columns:
            column = pandas.get_dummies(data=data_frame[c], prefix=c + '_')
            columns = pandas.concat((columns, column), axis=1)

        return columns
