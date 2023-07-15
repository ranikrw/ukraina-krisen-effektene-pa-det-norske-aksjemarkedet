import pandas as pd
import numpy as np
import datetime as dt

from functions_analyze import *

#######################################
## Define parameters
#######################################

# Define if running analyses per country or per industry
analyze_countries_or_industries = 'countries'
# 'countries'
# 'industries'

dict_parameters = {}

# Defining event date 24 February
dict_parameters['event_date'] = dt.date(2022, 2, 24)

# Days previous to event data that training data end
dict_parameters['days_end_training_data'] = 25

# How many days for OLS estimation period
dict_parameters['size_training_data'] = 250

dict_parameters['event_window_AR'] = [-3,3]

dict_parameters['event_windows_CAR'] = [
    [-25, 0],
    [-20, 0],
    [-15, 0],
    [-10, 0],
    [-5, 0],
    [-3, 0],
    [-2, 0],
    [-1, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [0, 5],
    [0, 10],
    [0, 15],
    [0, 20],
    [0, 25],
    [-1,1],
    [-2,2],
    [-3,3],
    [-5, 5],
    [-10, 10],
    [-15, 15],
    [-20, 20],
    [-25, 25],
]

# Making folder for saving data
folder_name_for_save = 'results_'+analyze_countries_or_industries
if not os.path.exists(folder_name_for_save):
    os.makedirs(folder_name_for_save)
dict_parameters['folder_name_for_save'] = folder_name_for_save

#######################################
## Analyzing
#######################################

if analyze_countries_or_industries == 'countries':
    indexes = pd.DataFrame([
        ['.TRXFLDNOT','Norway'], # Refinitiv Norway Total Return Index
        ['.OMXS30','Sweden'], # Sweden - OMX Stockholm 30 (OMXS30)
        ['.OMXC20','Denmark'], # OMX Copenhagen 20 (OMXC20)
        ['.OMXH25','Finland'], # OMX Helsinki 25 (OMXH25)
        ['.STOXX50E','Europe'], # EURO STOXX 50
    ],columns=['Index','Country'])
elif analyze_countries_or_industries == 'industries':
    indexes = pd.DataFrame([
        ['.TRXFLDNOTENE','Energy'], # Refinitiv Norway Energy Total Return Index
        ['.TRXFLDNOTIND','Industrials'], # Refinitiv Norway Industrials Total Return Index
        ['.TRXFLDNOTTEC','Technology'], # Refinitiv Norway Technology Total Return Index
        ['.TRXFLDNOTFIN','Financials'], # Refinitiv Norway Financials Total Return Index
        ['.TRXFLDNOTNCY','Consumer Non-Cyclicals'], # Refinitiv Norway Consumer Non-Cyclicals Total Return Index
    ],columns=['Index','Country'])


CAR_all = pd.DataFrame()
AAR_all = pd.DataFrame()
ARR_all_for_plotting = pd.DataFrame()
data_market_for_plotting_ALL = pd.DataFrame()
for RIC_index,country in zip(indexes['Index'],indexes['Country']):
    CAR_results,AAR_results,AAR,data_market_for_plotting = make_AR_and_CAR(RIC_index,country,dict_parameters)

    CAR_all = pd.concat([CAR_all,CAR_results],axis=1)

    AAR_all = pd.concat([AAR_all,AAR_results],axis=1)

    ARR_all_for_plotting = pd.concat([ARR_all_for_plotting,AAR],axis=1)

    data_market_for_plotting_ALL = pd.concat([data_market_for_plotting_ALL,data_market_for_plotting],axis=1)

AAR_all.to_excel(folder_name_for_save+'/ARR_all_'+analyze_countries_or_industries+'.xlsx')
CAR_all.to_excel(folder_name_for_save+'/CAR_all_'+analyze_countries_or_industries+'.xlsx')
data_market_for_plotting_ALL.to_csv(folder_name_for_save+'/market_data_for_plotting.csv',sep=';',index=True)
ARR_all_for_plotting.to_csv(folder_name_for_save+'/ARR_for_plotting.csv',sep=';',index=True)



