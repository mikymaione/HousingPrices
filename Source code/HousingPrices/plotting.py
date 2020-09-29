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

from typing import List

from dataTypes import DataElaboration


class Plotting:

    @staticmethod
    def plotNestedCrossVal(nested_cross_validation_trials, nested_scores, non_nested_scores, score_difference) -> None:
        # Plot scores on each trial for nested and non-nested CV
        plt.figure(figsize=(15, 15))
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

    @staticmethod
    def plotAreaMeanStd(title: str, x, y, neg: bool, labels: List[str], colors: List[str], xlabel: str, ylabel: str) -> None:
        y0 = y[0]
        y1 = y[1]

        if neg:
            m = -1
        else:
            m = 1

        y0_mean = m * numpy.mean(y0, axis=1)
        y0_std = numpy.std(y0, axis=1)

        y1_mean = m * numpy.mean(y1, axis=1)
        y1_std = numpy.std(y1, axis=1)

        Plotting.plotXYArea(
            title,
            x,
            [y0_mean, y1_mean],
            [y0_mean - y0_std, y1_mean - y1_std],
            [y0_mean + y0_std, y1_mean + y1_std],
            labels,
            colors,
            xlabel,
            ylabel)

    @staticmethod
    def plotAreaMeanStd1(title: str, x, y, neg: bool, labels: str, colors: str, xlabel: str, ylabel: str) -> None:
        if neg:
            m = -1
        else:
            m = 1

        y_mean = m * numpy.mean(y, axis=1)
        y_std = numpy.std(y, axis=1)

        plt.title(title)

        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1, color=colors)
        plt.plot(x, y_mean, color=colors, label=labels)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.grid()
        plt.legend()
        plt.show()

    @staticmethod
    def plotXYArea(title: str, x, y, a, b, labels: List[str], colors: List[str], xlabel: str, ylabel: str) -> None:
        # plt.figure(figsize=(15, 15))
        plt.title(title)

        for i in range(len(colors)):
            plt.fill_between(x, a[i], b[i], alpha=0.1, color=colors[i])
            plt.plot(x, y[i], color=colors[i], label=labels[i])
            # plt.plot(x, y[i], 'o-', color=colors[i], label=labels[i])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.grid()
        plt.legend()
        plt.show()

    @staticmethod
    def plotXY(x_serie, y_series, y_labels: List[str], xlabel: str, ylabel: str, title: str = '') -> None:
        # plt.figure(figsize=(15, 7))
        plt.title(title)

        for i in range(0, len(y_labels)):
            y_serie = y_series[i]
            plt.plot(x_serie, y_serie, label=y_labels[i])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot(title: str, x_serie, x_labels, xlabel: str = '', ylabel: str = '') -> None:
        # plt.figure(figsize=(15, 7))
        plt.title(title)

        plt.plot(x_serie, label=x_labels)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_DataElaboration(titolo: str, labels: List[str], P: DataElaboration) -> None:
        # plt.style.use('grayscale')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(titolo)
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

    @staticmethod
    def coeficientPlot(title: str, x_train, coef_) -> None:
        # get ridge coefficient and print them
        coefficient = pandas.DataFrame()
        coefficient["Columns"] = x_train.columns
        coefficient['Coefficient Estimate'] = pandas.Series(coef_)

        # plotting the coefficient score
        fig, ax = plt.subplots(figsize=(15, 7))
        # fig, ax = plt.subplots()
        ax.set_title(title)

        ax.bar(coefficient["Columns"], coefficient['Coefficient Estimate'])

        ax.spines['bottom'].set_position('zero')

        plt.grid()
        plt.show()

    @staticmethod
    def scatterPlot(x, y) -> None:
        # plt.figure(figsize=(15, 7))
        plt.grid()
        plt.scatter(x, y)
        plt.show()

    @staticmethod
    def regPlot(algo: str, y_predict, y_test) -> None:
        plt.figure(figsize=(15, 15))
        plt.title('Real vs Predicted - ' + algo)

        seaborn.regplot(y_test, y_predict, scatter_kws={'alpha': 0.5})

        plt.xlabel('Real')
        plt.ylabel('Predicted')

        plt.grid()
        plt.show()

    @staticmethod
    def heatMap(corr, title: str) -> None:
        plt.figure(figsize=(15, 15))
        plt.title(title)

        seaborn.heatmap(corr, square=True, annot=True)

        plt.show()

    @staticmethod
    def distPlot(data, title: str) -> None:
        plt.figure(figsize=(15, 7))
        plt.title(title)

        seaborn.distplot(data)

        plt.grid()
        plt.show()
