import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib import style
from pylab import polyfit
from pylab import poly1d
import seaborn as sns
import sklearn as skn
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pandas import DataFrame
import pickle
import random
import ModifyData


def scatter_plots(columnsNamesdf_scatter, df_scatter):
    xIterator = -1
    while xIterator < len(columnsNamesdf_scatter) - 2: # Using the -2 to only show the graphs in the bottom diagonal of the x by x grid of graphs
        xIterator = xIterator + 1
        x = df_scatter[columnsNamesdf_scatter[xIterator]]
        yIterator = len(columnsNamesdf_scatter)
        while yIterator > xIterator + 1:
            yIterator = yIterator - 1
            y = df_scatter[columnsNamesdf_scatter[yIterator]]
            fit = polyfit(x, y, 1)
            rSquared = round(np.corrcoef(x, y)[0, 1] * np.corrcoef(x, y)[0, 1], 3)
            fit_fn = poly1d(fit)
            plt.figure()
            plt.plot(x, y, '+', x, fit_fn(x), 'k', markersize=3)  # 'k' = black line
            plt.title(columnsNamesdf_scatter[xIterator + 1] + ' = ' + str("% .4e" % fit_fn[1]) + ' * ' + columnsNamesdf_scatter[
                xIterator] + ' + ' + str("% .4e" % fit_fn[0]))
            average = ModifyData.average_of_column(df_scatter, columnsNamesdf_scatter[yIterator])
            standard_deviation = ModifyData.standard_deviation_of_column(df_scatter, columnsNamesdf_scatter[yIterator])
            plt.suptitle('R squared = ' + str(rSquared) + ' Average ' + columnsNamesdf_scatter[yIterator] + '= ' + str(average) + ' Standard deviation = ' + str(standard_deviation))
            plt.xlabel(columnsNamesdf_scatter[xIterator])
            plt.ylabel(columnsNamesdf_scatter[yIterator])


def histograms(columnsNamesdf_histogram, df_histogram):
    xIterator = -1
    while xIterator < len(columnsNamesdf_histogram) - 1:
        xIterator = xIterator + 1
        plt.figure()
        df_histogram[columnsNamesdf_histogram[xIterator]].hist(bins=50)
        plt.xlabel(columnsNamesdf_histogram[xIterator])
        plt.title('Histogram')


def data_in_sequence(columnsNamesdf_sequence, df_sequence):
    xIterator = -1
    while xIterator < len(columnsNamesdf_sequence) - 1:
        xIterator = xIterator + 1
        plt.figure()
        plt.plot(df_sequence[columnsNamesdf_sequence[xIterator]], '+', markersize=3)
        plt.ylabel(columnsNamesdf_sequence[xIterator])
        plt.title('Sequential')

