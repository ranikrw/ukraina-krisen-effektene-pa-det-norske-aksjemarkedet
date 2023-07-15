import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm

import os

# Converting to log returns
def make_log_returns(data,instrument_to_make_log_return):
    log_returns = pd.Series([None]*data.shape[0],name=instrument_to_make_log_return,dtype=float,index=data.index)
    for i in range(1,data.shape[0]):
        log_returns.iloc[i] = 100*np.log(data[instrument_to_make_log_return].iloc[i]/data[instrument_to_make_log_return].iloc[i-1])
    data[instrument_to_make_log_return] = log_returns
    return data

def make_AR_and_CAR(RIC_index,country,dict_parameters):

    # Determining how many desimals when rounding for results tables
    num_decimal_round = 2

    #######################################
    ## Get parameter values
    #######################################
    event_date                  = dict_parameters['event_date']
    days_end_training_data      = dict_parameters['days_end_training_data']
    size_training_data          = dict_parameters['size_training_data'] 
    event_window_AR             = dict_parameters['event_window_AR']
    event_windows_CAR           = dict_parameters['event_windows_CAR']
    folder_name_for_save        = dict_parameters['folder_name_for_save']

    #######################################
    ## Load and prepare data
    #######################################
    folder_name             = 'data/'+RIC_index
    data_market             = pd.read_csv(folder_name+'/market.csv',sep=';',index_col=0)
    data_instruments_price  = pd.read_csv(folder_name+'/instruments_price.csv',sep=';',index_col=0)
    data_instruments_market_cap  = pd.read_csv(folder_name+'/instruments_market_cap.csv',sep=';',index_col=0)

    # Remove duplicates
    df = pd.DataFrame()
    df['index'] = data_market.index
    df[RIC_index] = list(data_market[RIC_index])
    df = df.drop_duplicates(subset=['index', RIC_index], keep='first')
    data_market = pd.DataFrame()
    data_market[RIC_index]  = list(df[RIC_index])
    data_market.index       = df['index']
    data_market.index.name = ''

    # Convert index format to date
    data_market.index                   = pd.Series(pd.to_datetime(data_market.index.tolist(), format='%Y-%m-%d')).dt.date
    data_instruments_price.index        = pd.Series(pd.to_datetime(data_instruments_price.index.tolist(), format='%Y-%m-%d')).dt.date
    data_instruments_market_cap.index   = pd.Series(pd.to_datetime(data_instruments_market_cap.index.tolist(), format='%Y-%m-%d')).dt.date

    #######################################
    ## Computing returns
    #######################################
    # Sorting by date
    data_market         = data_market.sort_index(ascending=True)
    data_instruments_price    = data_instruments_price.sort_index(ascending=True)

    # Making data later used for making plots
    data_market_for_plotting        = data_market.copy()
    data_market_for_plotting        = 100*data_market_for_plotting/data_market_for_plotting.iloc[0]
    data_market_for_plotting        = data_market_for_plotting.iloc[:,0]
    data_market_for_plotting.name   = country

    # Log returns - Market index
    data_market = make_log_returns(data_market,RIC_index)
    data_market = data_market.iloc[1:] # Remove first row

    # Log returns - Instruments
    for instrument in data_instruments_price.columns:
        data_instruments_price = make_log_returns(data_instruments_price,instrument)
    data_instruments_price = data_instruments_price.iloc[1:] # Remove first row

    #######################################
    ## Computing AR and AAR
    #######################################
    # Create data frame for AR
    AR_df = pd.DataFrame()

    # Compute AR
    for instrument in data_instruments_price.columns:
        data_instrument = data_instruments_price[instrument]

        # Drop missing
        data_instrument = data_instrument[pd.isnull(data_instrument)==False]

        # Define index of event date
        event_date_ind = data_instrument.index.get_loc(event_date)

        # Make training data
        last_day_training_data = event_date_ind-days_end_training_data
        temp = (last_day_training_data-size_training_data)
        data_training = pd.DataFrame(data_instrument.iloc[temp:last_day_training_data])
        data_training['Rm'] = data_market.loc[data_training.index]

        # Checking if correct size of training data
        if data_training.shape[0]!=size_training_data:
            print('{}({}): Size of trainng data is {}, and not {}, for instrument {}.'.format(country,RIC_index,data_training.shape[0],size_training_data,instrument))

        # Checking if missing values in training data
        if np.sum(np.sum(pd.isnull(data_training)))!=0:
            print('{}({}): ERROR missing values for data_training for instrument {}'.format(country,RIC_index,instrument))

        # Train model with OLS
        X = sm.add_constant(data_training['Rm'])
        r = data_training[instrument]
        model = sm.OLS(r, X).fit()
        # print(model.summary())

        X_test = data_market.loc[data_instrument.index]
        X_test.columns = ['Rm']
        X_test = sm.add_constant(X_test['Rm'])

        AR = data_instrument - model.predict(X_test)
        AR.name = instrument

        AR_df = pd.concat([AR_df,AR],axis=1)
    
    AAR = np.mean(AR_df,axis=1)

    # Define event window for AAR
    event_date_ind = AAR.index.get_loc(event_date)
    event_window_beginning = event_date_ind+event_window_AR[0]
    event_window_end = event_date_ind+event_window_AR[1]+1

    # Make AAR results
    AAR_results = pd.DataFrame()
    AAR_results['AAR']      = AAR.iloc[event_window_beginning:event_window_end]
    AAR_results.index = np.arange(event_window_AR[0],event_window_AR[1]+1)

    # Define event window for t-value calculations
    training_window_beginning = event_date_ind-days_end_training_data
    training_window_end = event_date_ind+days_end_training_data+1

    # Make t-values
    y = make_y_for_t_values(training_window_beginning,training_window_end,size_training_data,AAR,event_date)
    for i in range(event_window_AR[0],event_window_AR[1]+1):
        X = pd.Series(np.zeros(y.shape[0]),name='d',index=y.index)
        X.loc[i]=1
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        # print(model.summary())

        # Formatting AAR
        AAR_results.at[i,'AAR']     = format_val(AAR_results.at[i,'AAR'],num_decimal_round)
        tverdi                      = format_val(model.tvalues['d'],num_decimal_round)
        AAR_results.at[i,'AAR']     = AAR_results.at[i,'AAR'] + '(' + tverdi + ')'

    #######################################
    ## Computing CAR
    #######################################

    # Make AAR table for calculating CAR
    AAR_for_CAR = pd.DataFrame()
    AAR_for_CAR['AAR']      = AAR.iloc[training_window_beginning:training_window_end]
    AAR_for_CAR.index = np.arange(-days_end_training_data,days_end_training_data+1)

    # Create data frame for CAR
    CAR_results = pd.DataFrame()
    for i in event_windows_CAR:
        temp = pd.DataFrame()
        temp['Begin']   = [i[0]]
        temp['End']     = [i[1]]
        temp['Event Windows'] = ['['+str(i[0])+','+str(i[1])+']']
        CAR_results = pd.concat([CAR_results,temp],axis=0)
    CAR_results['CAR'] = pd.Series([None]*CAR_results.shape[0])
    CAR_results = CAR_results.reset_index(drop=True)

    # Preparing data for making t-values
    y = make_y_for_t_values(training_window_beginning,training_window_end,size_training_data,AAR,event_date)

    for i in CAR_results.index:
        Begin   = CAR_results.loc[i]['Begin']
        End     = CAR_results.loc[i]['End']
        CAR_results.at[i,'CAR'] = format_val(np.sum(AAR_for_CAR.loc[Begin:End]).iloc[0],num_decimal_round)
        
        # t-value
        X = pd.Series(np.zeros(y.shape[0]),name='d',index=y.index)
        X.loc[Begin:End]=1
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        # print(model.summary())

        tverdi = format_val(model.tvalues['d'],num_decimal_round)
        CAR_results.at[i,'CAR'] = CAR_results.at[i,'CAR']+'('+tverdi+')'

    #######################################
    ## Make final data frame
    #######################################
    del CAR_results['Begin']
    del CAR_results['End']

    CAR_results.columns = [
        CAR_results.columns[0],
        CAR_results.columns[1]+' '+country+' ('+RIC_index+')',
        ]

    AAR_results.columns = [
        AAR_results.columns[0]+' '+country+' ('+RIC_index+')',
        ]

    CAR_results.index = CAR_results['Event Windows'].values
    del CAR_results['Event Windows']

    event_date_ind = AAR.index.get_loc(event_date)
    AAR = AAR.reset_index(drop=True)
    AAR.index=AAR.index-event_date_ind
    AAR.name=country

    return CAR_results,AAR_results,AAR,data_market_for_plotting


def format_val(val,num_decimal_round):
    string = str(np.round(val,num_decimal_round))
    if string[-(num_decimal_round+1)]!='.':
        string = string+'0'
    return string.replace('.', ',')

def make_y_for_t_values(training_window_beginning,training_window_end,size_training_data,AAR,event_date):
    temp = training_window_beginning-size_training_data
    if temp<0:
        print('WARNING: Missing {} trading days for calculating t-values'.format(np.abs(temp)))
        temp = 0
    y = AAR.iloc[temp:training_window_end]
    event_date_ind = y.index.get_loc(event_date)
    y = y.reset_index(drop=True)
    y.index=y.index-event_date_ind
    return y