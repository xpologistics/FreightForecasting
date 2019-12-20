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


def filter_dataframe(dataframe, column_to_filter, filter_type, value_restriction):
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
        dataframe[column_name_is_alphanumeric] = dataframe[column_name].apply(lambda x: True if str(x).isalnum() else False)
        column_name_is_alpha = column_name + "_is_alpha"
        dataframe[column_name_is_alpha] = dataframe[column_name].apply(lambda x: True if str(x).isalpha() else False)
        column_name_is_digit = column_name + "_is_digit"
        dataframe[column_name_is_digit] = dataframe[column_name].apply(lambda x: True if str(x).isdigit() else False)
    return dataframe


def average_of_column(dataframe, column_to_average):
    return round(dataframe[column_to_average].mean(), 2)


def standard_deviation_of_column(dataframe, column_to_average):
    return round(dataframe[column_to_average].std(), 2)


def standard_deviations_from_mean(mean, std_dev, value):
    return abs(mean - value) / std_dev


def outlier_calculation(dataframe, dataframe_column_name, outlier_sigma_count=3):
    standard_deviation_test = standard_deviation_of_column(dataframe, dataframe_column_name)
    average_test = average_of_column(dataframe, dataframe_column_name)
    column_name_outliers = dataframe_column_name + "_outlier"
    dataframe[column_name_outliers] = dataframe.apply(lambda row: abs(standard_deviations_from_mean(average_test, standard_deviation_test, row[dataframe_column_name])) >= outlier_sigma_count, axis = 1)
    #dataframe_outliers = np.DataFrame()
    #dataframe_outliers[dataframe_column_name] = dataframe[column_name_outliers].apply(lambda row: dataframe[dataframe_column_name] if dataframe[column_name_outliers] else 0)
    return dataframe


