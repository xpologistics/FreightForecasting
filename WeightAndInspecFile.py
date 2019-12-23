import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import PreparePlots

scatterPlot = "true"

# Read in the complete data set
#df_Train = pd.read_csv("C:\\Users\\baker.todd\\Documents\\WnI_Lineal.csv")
#df_Train = pd.read_csv("C:\\Users\\baker.todd\\Documents\\WnI_Lineal_RAW.csv")
df_Train = pd.read_table("E:\\conway\\WNI\\2019\\Data\\Lineal_Model\\ANNLineal.txt", delim_whitespace=True)
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
    df_plots = df_Train.loc[:, ['SHIPMENT_VOLUMNE_CUBIC_FOOT','SIC_MILES','CNT_OF_SHIPMENTS_PA','MOTORIZED_PIECES_COUNT','SHIPMENT_WEIGHT','CNT_OF_CORRECTIONS_PA','SHIPPER_LATD','DENSITY','CORR_REVENUE_PA','PA_NAICS6','CONS_LATD','SHPMT_CORR_REV']]
    #df_plots = df_plots.sort_values(by='AMC_IND')  # print(df_scatter)
    # Super plot
    #sns.pairplot(df_plots, diag_kind="kde", markers="+",
    #             plot_kws=dict(s=10, edgecolor="b", linewidth=1),
    #             diag_kws=dict(
    #                 shade=True))  # kde = gausian Kernel density estimation of histogram, smaller dots
    #df_plots = df_Train
    df_plots = df_plots.sample(n=100000) # plots get overloaded with 13MM rows
    columnsNamesdf_plots = df_plots.columns.values  # print(columnsNamesdf_scatter)
    # scatter plots
    PreparePlots.scatter_plots(columnsNamesdf_plots, df_plots)
    # Histograms
 #   PreparePlots.histograms(columnsNamesdf_plots, df_plots)
    # data in sequence
 #   PreparePlots.data_in_sequence(columnsNamesdf_plots, df_plots)
    plt.show()