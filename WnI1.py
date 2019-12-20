# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:53:51 2019

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
import ModifyData

scatterPlot = "true"
# Read in the complete data set
#df_Train = pd.read_csv("C:\\Conway\\Unprojected_Freight\\2019\\lho_shipment_inpt.csv")
df_Train = pd.read_csv("E:\\Conway\\WNI\\2019\\Data\\Lineal_Model\\WnI_Lineal_MULTI.csv")
df_Train['Count'] = np.arange(len(df_Train))                # add a count column 0,1,2,3,...
#print(df_Train)#
columnsNamesArr = df_Train.columns.values   
print(columnsNamesArr)
print('df_Train pre filter')
print(df_Train.shape)
# Filter data by value
#isdf_fltr = df_Train["SHIPPER_LNGT"] <= -60.               # No cube % > 100%
#df_Train = df_Train[isdf_fltr]
df_Train = df_Train[ModifyData.filter_dataframe(df_Train, "SHIPPER_LNGT", "<=", -60)]
isdf_fltr = df_Train["SHIPPER_LNGT"] >= -125.  
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SHIPPER_LATD"] <= 50          
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SHIPPER_LATD"] >= 25.       
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["CONS_LATD"] <= 50.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["CONS_LATD"] >= 25.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["CONS_LNGT"] <= -60.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["CONS_LNGT"] >= -125.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SHPMT_CORR_REV"] >= 0.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["DENSITY"] <= 150.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["DENSITY"] > 0.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SHIPMENT_VOLUMNE_CUBIC_FOOT"] <= 5000.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SHIPMENT_VOLUMNE_CUBIC_FOOT"] >  0.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["LOOSE_PCE_CNT"] <= 100.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["LOOSE_PCE_CNT"] >= 0.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["MOTORIZED_PIECES_COUNT"] <= 60.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["MOTORIZED_PIECES_COUNT"] >= 0.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SIC_MILES"] <= 3500.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SIC_MILES"] >= 0.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SHIPMENT_WEIGHT"] <= 15000.
df_Train = df_Train[isdf_fltr]
isdf_fltr = df_Train["SHIPMENT_WEIGHT"] > 0.
df_Train = df_Train[isdf_fltr]
#Hard Limit
df_Train.loc[df_Train.CORR_REVENUE_PA > 500000, 'CORR_REVENUE_PA'] = 500000
df_Train.loc[df_Train.CNT_OF_CORRECTIONS_PA > 500, 'CNT_OF_CORRECTIONS_PA'] = 500
df_Train.loc[df_Train.CNT_OF_SHIPMENTS_PA > 100000, 'CNT_OF_SHIPMENTS_PA'] = 100000
df_Train.loc[df_Train.CORR_REVENUE_SHPR > 100000, 'CORR_REVENUE_SHPR'] = 100000
df_Train.loc[df_Train.CNT_OF_CORRECTIONS_SHPR > 100, 'CNT_OF_CORRECTIONS_SHPR'] = 100
df_Train.loc[df_Train.CNT_OF_SHIPMENTS_SHPR > 12000, 'CNT_OF_SHIPMENTS_SHPR'] = 12000
df_Train.loc[df_Train.SHIPMENT_VOLUMNE_CUBIC_FOOT > 2000, 'SHIPMENT_VOLUMNE_CUBIC_FOOT'] = 2000
df_Train.loc[df_Train.LOOSE_PCE_CNT > 20, 'LOOSE_PCE_CNT'] = 20
df_Train.loc[df_Train.MOTORIZED_PIECES_COUNT > 25, 'MOTORIZED_PIECES_COUNT'] = 25
df_Train.loc[df_Train.SIC_MILES > 3000, 'SIC_MILES'] = 3000
df_Train.loc[df_Train.SHIPMENT_WEIGHT > 10000, 'SHIPMENT_WEIGHT'] = 10000
df_Train.loc[df_Train.DENSITY > 60, 'DENSITY'] = 60
# remaining data
print('df_Train post filter')
print(df_Train.shape) 

if scatterPlot == "true":
    #select data for the plots
    #df_scattertmp = df_Train.loc[:, [' SHIPMENT_VOLUMNE_CUBIC_FOOT',  ' DENSITY',   ' SHPMT_CORR_REV']] 
    df_scattertmp = df_Train.loc[:, :] 
    #df_scatter = df_scattertmp.sort_values(by='PKUP_DT')    #print(df_scatter)    
    df_scatter = df_scattertmp 
    # Super plots
    #sns.pairplot(df_scatter, diag_kind="kde", markers="+",
    #              plot_kws=dict(s=10, edgecolor="b", linewidth=1),
    #              diag_kws=dict(shade=True))                # kde = gausian Kernel density estimation of histogram, smaller dots                                                                   
    columnsNamesdf_scatter = df_scatter.columns.values      #print(columnsNamesdf_scatter)
    # scatter plots
    ivv = 0
    #while ivv < len(columnsNamesdf_scatter) - 2:
    #    ivv = ivv + 1
    #    x = df_scatter[columnsNamesdf_scatter[ivv]] 
    #    y = df_scatter[columnsNamesdf_scatter[ivv+1]]
    #    fit = polyfit(x, y, 1)
    #    fit_fn = poly1d(fit)
    #    plt.figure()
    #    plt.plot(x,y, '+', x, fit_fn(x), 'k', markersize=3)               # 'k' = black line
    #    plt.title(columnsNamesdf_scatter[ivv+1] + ' = ' + str("% .4e" %fit_fn[1]) + ' * ' + columnsNamesdf_scatter[ivv] + ' + ' + str("% .4e" %fit_fn[0]) )
    #    plt.xlabel(columnsNamesdf_scatter[ivv])
    #    plt.ylabel(columnsNamesdf_scatter[ivv+1])        
    #    plt.show()  
    # Histograms
    ivv = 0
    while ivv < len(columnsNamesdf_scatter) - 1:
        ivv = ivv + 1
        plt.figure()
        df_scatter[columnsNamesdf_scatter[ivv]].hist(bins=50) 
        plt.title(columnsNamesdf_scatter[ivv])                   
        plt.xlabel(columnsNamesdf_scatter[ivv])
        plt.show() 
    # by row
    plt.figure()
    plt.plot(df_scatter['SHPMT_CORR_REV'],'+',markersize=3 )
    plt.title('SHPMT_CORR_REV')
    plt.ylabel('SHPMT_CORR_REV')
    plt.show()
    #ivv = 0
    #while ivv < len(columnsNamesdf_scatter) - 1:
    #    ivv = ivv + 1
    #    plt.figure()
    #    plt.plot(df_scatter[columnsNamesdf_scatter[ivv]],'+',markersize=3 )
    #    plt.title(columnsNamesdf_scatter[ivv])
    #    plt.ylabel(columnsNamesdf_scatter[ivv])
    #    plt.show()
    
    # Print to a file
    # without header line
    np.savetxt(r'E:\\Conway\\WNI\\2019\\Data\\Lineal_Model\\ANNLineal.txt', df_Train.values, fmt='%10.3e')
    
    

                             
 