# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:24:39 2019

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
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import pickle
import random
import ModifyData
import PreparePlots
#from mlxtend.plotting import plot_linear_regression

# Read in the complete data set
df_Train = pd.read_table("C:\\conway\\WNI\\2019\\Data\\Lineal_Model\\ANNLineal.txt", delim_whitespace=True)
print('df_Train.shape')
print(df_Train.shape)
columnsNamesdf_Train = df_Train.columns.values
print(columnsNamesdf_Train)

# make training set and test set 
# random samples
data1 = df_Train
data1['Count'] = np.arange(len(data1))  # add a count column 0,1,2,3,...
X = data1.iloc[:,0:29]
print('X.shape')
print(X.shape)
y = data1.iloc[:,29:30]
print('y.shape')
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.90, random_state=42)
print('X_train.shape')
print(X_train.shape)
print('X_test.shape')
print(X_test.shape)
print('y_train.shape')
print(y_train.shape)
print('y_test.shape')
print(y_test.shape)

# Train Models
regr = skn.linear_model.LinearRegression()      # Create linear regression object
regr.fit(X_train, y_train)

# Test Models
regr_model_InSample = regr.predict(X_train)

#plot
x1 = y_train['SHPMT_CORR_REV']
y1 = regr_model_InSample
plt.figure()
plt.scatter(x1, y1,  color='black')
plt.show()  


