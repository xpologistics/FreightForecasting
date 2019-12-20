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
import PreparePlots

scatterPlot = "false"
scatterPlot = "true"

# Read in the complete data set
#df_Train = pd.read_csv("C:\\Users\\baker.todd\\Documents\\WnI_Lineal.csv")
df_Train = pd.read_csv("C:\\Users\\baker.todd\\Documents\\WnI_Lineal_RAW.csv")
# df_Train = pd.read_csv("E:\\Conway\\Unprojected_Freight\\2019\\lho_shipment_inpt.csv")
df_Train['Count'] = np.arange(len(df_Train))  # add a count column 0,1,2,3,...
# print(df_Train)#columnsNamesArr = df_Train.columns.values#print(columnsNamesArr)

print('df_Train pre filter')
print(df_Train.shape)

df_Train_column_names = df_Train.columns.values

#df_Train = ModifyData.add_numeric_alpha_alphanum_tests(df_Train, df_Train_column_names)

print('df_Train post filter')
print(df_Train.shape)



if scatterPlot == "true":
    # select data for the plot
    df_plotstmp = df_Train.loc[:, [' SHPMT_CORR_REV', ' AMC_IND', ' DENSITY']]
    df_plots = df_plotstmp.sort_values(by=' AMC_IND')  # print(df_scatter)
    # Super plot
    sns.pairplot(df_plots, diag_kind="kde", markers="+",
                 plot_kws=dict(s=10, edgecolor="b", linewidth=1),
                 diag_kws=dict(
                     shade=True))  # kde = gausian Kernel density estimation of histogram, smaller dots
    columnsNamesdf_plots = df_plots.columns.values  # print(columnsNamesdf_scatter)
    df_outliers = ModifyData.outlier_calculation(df_plots, ' DENSITY')
    print(df_outliers[' DENSITY_outlier'].min())
    # scatter plots
    xIterator = -1
    PreparePlots.scatter_plots(xIterator, columnsNamesdf_plots, df_plots)
    # Histograms
    xIterator = -1
    PreparePlots.histograms(xIterator, columnsNamesdf_plots, df_plots)
    # data in sequence
    xIterator = -1
    PreparePlots.data_in_sequence(xIterator, columnsNamesdf_plots, df_plots)
    plt.show()