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


def filter_dataframe(dataframe, column_to_filter, value_restriction, filter_type):
    if filter_type == "<=":
        dataframe = dataframe[column_to_filter] <= value_restriction
    elif filter_type == ">=":
        dataframe = dataframe[column_to_filter] >= value_restriction
    elif filter_type == "<":
        dataframe = dataframe[column_to_filter] < value_restriction
    elif filter_type == ">":
        dataframe = dataframe[column_to_filter] > value_restriction
    return dataframe


def add_numeric_alpha_alphanum_tests(dataframe, dataframe_column_names):
    for column_name in dataframe_column_names:
        column_name_is_alphanumeric = column_name + "_is_alphanumeric"
        dataframe[column_name_is_alphanumeric] = np.where(str(dataframe[column_name].values).isalnum(), True, False)
        column_name_is_alpha = column_name + "_is_alpha"
        dataframe[column_name_is_alpha] = np.where(str(dataframe[column_name].values).isalpha(), True, False)
        column_name_is_digit = column_name + "_is_digit"
        dataframe[column_name_is_digit] = np.where(str(dataframe[column_name].values).isdigit(), True, False)
    return dataframe
