# PRO_NBR_TXT
# SHP_INST_ID
# PUR_INST_ID
# PUR_SEQ_NBR
# PKUP_DT
# ORIG_SIC_CD
# ORIG_HOST_SIC_CD
# DEST_SIC_CD
# DEST_HOST_SIC_CD
# CURR_SIC_CD
# CURR_TRLR_ID
# HEAD_LD_IND
# WGT_LBS
# CUBE_PCT
# FRZBLE_IND
# HAZMAT_IND
# ESTD_DLVY_DT
# CNFM_ONBRD_IND
# PKUP_TERM_SIC_CD
# PKUP_CALL_DT
# PKUP_CALL_SEQ_NBR
# MISLOAD_IND
# DIVERTABLE_IND
# BULK_LQD_IND
# COBZ_IND
##Origin=current => outbound	one every hour = forecasting cube, shipmenbts , by hour for outbound

# Above: input variables by column number
# ========================================

# ------------------------------------------------------------------------------
# NOTES
# HOW TO RUN DNAF, GRNNS , size clustering, From PYTHON ???????
# I WILL NOT (!!!) SET A LEARNING RATE SCHEDULE !!!!!!!
# and I NEED a size clustering algo.
# shell call ?
# txt file interface?
# plots
# ------------------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""
Created
 -- Nov 2019.
@author: bartlett.eric
"""

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

scatterPlot = "false"
scatterPlot = "true"

# Read in the complete data set
df_Train = pd.read_csv("C:\\Users\\baker.todd\\Documents\\lho_shipment_inpt.csv")
# df_Train = pd.read_csv("E:\\Conway\\Unprojected_Freight\\2019\\lho_shipment_inpt.csv")
df_Train['Count'] = np.arange(len(df_Train))  # add a count column 0,1,2,3,...
# print(df_Train)#columnsNamesArr = df_Train.columns.values#print(columnsNamesArr)

print('df_Train pre filter')
print(df_Train.shape)

# add density
isdf_fltrx = df_Train["CUBE_PCT"] >= 0.00001  # No cube % < .00001%
df_Train = df_Train[isdf_fltrx]
df_Train['Density'] = df_Train["WGT_LBS"] / df_Train["CUBE_PCT"]  # add a column Density

# Filter data by value
isdf_fltr = df_Train["CUBE_PCT"] <= 120.  # No cube % > 100%
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["WGT_LBS"] <= 23000.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["WGT_LBS"] >= 1.0
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["PUR_SEQ_NBR"] <= 15.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["Density"] <= 2500.
df_Train = df_Train[isdf_fltr]

print('df_Train post filter')
print(df_Train.shape)

if scatterPlot == "true":
    # select data for the plot
    df_scattertmp = df_Train.loc[:, ['PKUP_DT', 'WGT_LBS', 'CUBE_PCT', 'Density', 'PUR_SEQ_NBR']]
    df_scatter = df_scattertmp.sort_values(by='PKUP_DT')  # print(df_scatter)
    # Super plot
    sns.pairplot(df_scatter, diag_kind="kde", markers="+",
                 plot_kws=dict(s=10, edgecolor="b", linewidth=1),
                 diag_kws=dict(
                     shade=True))  # kde = gausian Kernel density estimation of histogram, smaller dots
    columnsNamesdf_scatter = df_scatter.columns.values  # print(columnsNamesdf_scatter)
    # scatter plots
    xIterator = 0
    while xIterator < len(columnsNamesdf_scatter) - 2:
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
            plt.title('Sequential Histogram ' + columnsNamesdf_scatter[xIterator + 1] + ' = ' + str("% .4e" % fit_fn[1]) + ' * ' + columnsNamesdf_scatter[
                xIterator] + ' + ' + str("% .4e" % fit_fn[0]))
            plt.suptitle('R squared = ' + str(rSquared))
            plt.xlabel(columnsNamesdf_scatter[xIterator])
            plt.ylabel(columnsNamesdf_scatter[yIterator])
        # plt.show()
    # Histograms
    xIterator = 0
    while xIterator < len(columnsNamesdf_scatter) - 1:
        xIterator = xIterator + 1
        plt.figure()
        df_scatter[columnsNamesdf_scatter[xIterator]].hist(bins=50)
        plt.xlabel(columnsNamesdf_scatter[xIterator])
        # plt.show()
    # data in sequence
    xIterator = 0
    while xIterator < len(columnsNamesdf_scatter) - 1:
        xIterator = xIterator + 1
        plt.figure()
        plt.plot(df_scatter[columnsNamesdf_scatter[xIterator]], '+', markersize=3)
        plt.ylabel(columnsNamesdf_scatter[xIterator])
        # plt.show()
    plt.show();
else:

    uniquedf = df_Train["ORIG_SIC_CD"].unique()  # find unique SICs
    # for each unique ORIG_SIC_CD
    iiiv = -1  # row 0 is the title line
    while iiiv < len(uniquedf) - 1:  # -1 // len is NOT zero indexed
        iiiv = iiiv + 1
        # Find the O-C pairs Data
        isLWLdf_fltr1t = df_Train["ORIG_SIC_CD"] == uniquedf[iiiv]
        df_fltr1t = df_Train[isLWLdf_fltr1t]  # is a unique SIC == myCsic
        # print(df_fltr1t)
        #
        df_fltr1 = df_fltr1t
        #
        #     isLWLdf_fltr1 = df_fltr1t["CURR_SIC_CD"] == uniquedf[iiiv]
        # print(isLWLdf_fltr1)
        #     df_fltr1 = df_fltr1t[isLWLdf_fltr1]                 # df_fltr1 contains only the O-C pair data # Unsorted
        # print(df_fltr1["PKUP_DT"])

        # Sort assending by date
        df_fltr2 = df_fltr1.sort_values(by='PKUP_DT')  # Sort by Date
        df_fltr2['Count'] = np.arange(
            len(df_fltr2))  # add a column row index, Find how many rows for this O-C pair#print(df_fltr2.count())

        # Find sum(CUBE_PCT per day) # for each date - do a group sum
        df_sumCUBE = df_fltr2.groupby(['PKUP_DT']).sum()  # only summable columns are transfered !!!!
        df_sumCUBE['Count'] = np.arange(len(df_sumCUBE))  # add a column row index

        if len(df_sumCUBE) > 3:
            # Train the model using the training sets
            df_in = df_sumCUBE.iloc[:, 8].values.reshape(-1,
                                                         1)  # column 9-1 = 8 = "Count" # [:,0 ] is all the rows from the zeroth (first) column
            df_out = df_sumCUBE.iloc[:, 6].values.reshape(-1, 1)  # column 7-1 = 6 = "CUBE_PCT"

            # add 300 random rows to each
            maxElement = np.amax(df_out)  # Get the maximum element from a Numpy array
            minElement = np.amin(df_out)
            maxIn = np.amax(df_in)  # print(minElement) #print(maxElement)
            for x in range(300):
                RndInt = random.randint(int(minElement), int(maxElement))
                df_out = np.append(df_out, RndInt)
                maxIn = maxIn + 1
                df_in = np.append(df_in, maxIn)  # print(df_out) #print(RndInt)

            # make them readabble by skn.linear_model.LinearRegression()
            df_in = DataFrame(df_in, columns=['x_test_values'])  # a dataframe for input to regression
            df_in = df_in.iloc[:, 0].values.reshape(-1,
                                                    1)  # a "2-D" single column of inputs for input to regression
            df_out = DataFrame(df_out, columns=['x_test_values'])  # a dataframe for input to regression
            df_out = df_out.iloc[:, 0].values.reshape(-1,
                                                      1)  # a "2-D" single column of inputs for input to regression
            # print(df_in) #print(df_out)

            # fit each model
            # regr = skn.linear_model.LinearRegression()      # Create linear regression object
            # regr.fit(df_in, df_out)
            df_outRF = np.ravel(df_out)  # # make it readabble by RandomForestRegressor
            clf_RandomForestRegressor = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_split=10,
                                                              random_state=None,
                                                              oob_score="true", n_jobs=1, verbose=0)
            regr = clf_RandomForestRegressor.fit(df_in, df_outRF)

            # save each model
            pkl_filename = "regr_model_" + str(
                iiiv) + ".pkl"  # Save to file in the current working directory
            with open(pkl_filename, 'wb') as file:
                pickle.dump(regr, file)
                score = regr.score(df_in,
                                   df_out)  # The .score method of the LinearRegression returns the coefficient of determination R^2 of the prediction not the accuracy
                print("In-Sample R^2: {0:.2f} %".format(100 * score))

            # make the testing input data for out-of-sample
            # note the input is just the count number
            dfMax = df_in.max() + 1  # highest value in data set dfMax = an int
            in_testtmp = [dfMax]  # in_testtmp = a list: 'array([304])'
            for x in range(10):
                dfMax = dfMax + 1
                in_testtmp = np.append(in_testtmp,
                                       dfMax)  # in_testtmp = a list: 'array([304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314])'

            # make them readabble by skn.linear_model.LinearRegression()
            df_in_testtmp = DataFrame(in_testtmp, columns=['x_test_values'])  # a dataframe for input to regression
            df_in_test = df_in_testtmp.iloc[:, 0].values.reshape(-1,
                                                                 1)  # a "2-D" single column of inputs for input to regression   #print(df_in_test)

            # READ in the model
            with open(pkl_filename, 'rb') as file:  # Load model from file
                regr_model = pickle.load(file)

                # predict in-sample
            regr_model_InSample = regr_model.predict(df_in)

            # Forecast out-of-sample
            regr_model_OutSample = regr_model.predict(df_in_test)

            # plot results in sequence
            pl.figure()  # set figure (move) to front (set focus)
            pl.plot(df_in, df_out, label='Actual')
            pl.plot(df_in, regr_model_InSample, label='Model')
            pl.plot(df_in_test, regr_model_OutSample, label='Prediction')
            pl.legend(loc='upper left')
            pl.xlabel("Dates")
            pl.ylabel("CUBE_PCT")
            pl.title(" ORIG_SIC_CD = " + uniquedf[iiiv])
            pl.show()
            cfm = pl.get_current_fig_manager()  # set figure to front
            cfm.window.activateWindow()  # set figure to front
            cfm.window.raise_()  # set figure to front

            # plot results by scatter Actual & Model
            x = np.ravel(df_in)
            y = np.ravel(df_out)
            fit = polyfit(x, y, 1)
            fit_fn = poly1d(fit)
            plt.figure()
            plt.plot(x, y, '+', x, fit_fn(x), 'k', markersize=3)  # 'k' = black line
            plt.title(
                "CUBE_PCT" + ' = ' + str("% .4e" % fit_fn[1]) + ' * ' + 'Count' + ' + ' + str("% .4e" % fit_fn[0]))
            # need R^2  print here with "In - sample"
            plt.suptitle("ORIG_SIC_CD = " + uniquedf[iiiv] + ' :  In - Sample  R^2 = ' + str("% 6.3f" % score))
            plt.ylabel("CUBE_PCT")
            plt.show()

            # Plot Error distributions
            # error_InSample = regr_model_InSample - df_in
            # print('bias')
            # print(avg(error_InSample))

################################################################################
# myDates = df_fltr3["PKUP_DT"].value_counts()        # find dates & the count of rows for each date => myDates
# print(myDates.keys())                               # Print only the dates = keys
# uDates = df_fltr3["PKUP_DT"].nunique()              # Find number of unique dates for this O-C pair = uDates #print(uDates)
# print(df_s2['Count'])
# Get ndArray of all column names
# columnsNamesArr = df_s2.columns.values
# print(columnsNamesArr)

# prediction - IN-SAMPLE
# prediction WITHOUT reading the model from a file
# df_s2y_pred = regr.predict(df_in)
# Forecast sample & t+n
# df_s2y_pred = regr.predict(df_in_test)
# print(df_s2y_pred)
# df_s2y_Mod = regr.predict(df_in)
# sns.pairplot(df_scatter, kind = "reg")                 # VERY SLOW !!!
# sns.pairplot(df_scatter,  markers="+", plot_kws=dict(s=10)) # histograms by default
# g = sns.pairplot(df_scatter, kind="reg", plot_kws={'line_kws':{'color':'red'}})  #SLOW, no equations
# g.show()
# a = pl.figure(0)
# print(df_s2["CUBE_PCT"])
# pl.plot(df_s2["CUBE_PCT"])
# pl.title(" ORIG_SIC_CD = " + myCsic + ": CURR_SIC_CD = " + myCsic)
# pl.xlabel("PKUP_DT")
# pl.ylabel("CUBE_PCT")
# pl.pause(2)
# pl.close()