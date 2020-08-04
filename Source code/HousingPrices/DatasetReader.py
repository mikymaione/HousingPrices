# MIT License
#
# Copyright (c) 2020 Anna Olena Zhab'yak, Michele Maione
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Here is a description of the attributes in the dataset:
# 0. longitude: A measure of how far west a house is; a higher value is farther west
# 1. latitude: A measure of how far north a house is; a higher value is farther north
# 2. housingMedianAge: Median age of a house within a block; a lower number is a newer building
# 3. totalRooms: Total number of rooms within a block
# 4. totalBedrooms: Total number of bedrooms within a block
# 5. population: Total number of people residing within a block
# 6. households: Total number of households, a group of people residing within a home unit, for a block
# 7. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# 8. medianHouseValue: Median house value for households within a block (measured in US Dollars)
# 9. oceanProximity: Location of the house w.r.t ocean/sea
#
# Note: The dataset has an attribute with missing values and an attribute with categorical values. Find a way of handling these anomalies and justify your choice.
import csv
import math
import numpy as np


class DatasetReader:

    def NumberOfLines(self, filePath):
        """Return number of lines of file.

        Parameters
        ----------
        filePath : the path of the file on the disk

        Returns
        -------
        tot : the number of lines
        """
        tot = 0

        with open(filePath) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for row in reader:
                tot += 1

        return tot

    def Read(self, filePath, trainingPct):
        """Read the dataset.

        Parameters
        ----------
        filePath : the path of the file on the disk
        trainingPct : the % of training set / test set

        Returns
        -------
        train_S : the training set without medianHouseValue
        train_R : the training set medianHouseValue
        test_S : the test set without medianHouseValue
        test_R : the test set medianHouseValue
        """

        assert trainingPct <= 100
        assert trainingPct >= 0

        tot = self.NumberOfLines(filePath) - 1
        train = math.ceil(tot * trainingPct / 100)
        test = tot - train

        curr_train = -1
        curr_test = -1

        train_S = np.zeros((train, 9), dtype='f')
        train_R = np.zeros((train, 1), dtype='f')
        test_S = np.zeros((test, 9), dtype='f')
        test_R = np.zeros((test, 1), dtype='f')

        with open(filePath) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

            next(reader)  # salta header

            for row in reader:
                oceanProximity = row.pop()
                medianHouseValue = row.pop()

                row.append(self.oceanProximityToNumber(oceanProximity))

                for e in range(0, 9):
                    if row[e] == '':
                        row[e] = 0

                if train > 0:
                    curr_train += 1
                    train_S[curr_train] = row
                    train_R[curr_train] = medianHouseValue
                else:
                    curr_test += 1
                    test_S[curr_test] = row
                    test_R[curr_test] = medianHouseValue

                train -= 1

        return train_S, train_R, test_S, test_R

    def oceanProximityToNumber(self, oceanProximity):
        switcher = {
            "ISLAND": 500,
            "NEAR BAY": 100,
            "NEAR OCEAN": 50,
            "<1H OCEAN": 20,
            "INLAND": 1,
        }

        return switcher.get(oceanProximity)
