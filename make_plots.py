import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick # For formatting axis with percent

from dateutil.relativedelta import relativedelta

#######################################
## Define parameters
#######################################

# Define if running analyses per country or per industry
analyze_countries_or_industries = 'countries'
# 'countries'
# 'industries'

# Defining event date 24 February
event_date = dt.date(2022, 2, 24)

#######################################
## Load data
#######################################
folder_name = 'results_'+analyze_countries_or_industries

data_market_for_plotting    = pd.read_csv(folder_name+'/market_data_for_plotting.csv',sep=';',index_col=0)
ARR_for_plotting            = pd.read_csv(folder_name+'/ARR_for_plotting.csv',sep=';',index_col=0)

# Convert index format to date
data_market_for_plotting.index    = pd.Series(pd.to_datetime(data_market_for_plotting.index.tolist(), format='%Y-%m-%d')).dt.date

# Restricting the period
data_market_for_plotting = data_market_for_plotting[data_market_for_plotting.index<=dt.date(2022, 8, 31)]

# Sorting by date
data_market_for_plotting         = data_market_for_plotting.sort_index(ascending=True)

#######################################
## Rename columns
#######################################
def rename_columns(df):
    return df.rename(columns = {
    'Norway':'Norge',
    'Sweden':'Sverige',
    'Denmark':'Danmark',
    'Finland':'Finland',
    'Europe':'Europa',
    'Energy':'Energi',
    'Industrials':'Industri',
    'Technology':'Teknologi',
    'Financials':'Finans',
    'Consumer Non-Cyclicals':'Ikke-sykliske forbruksvarer',
    }, inplace = False)

data_market_for_plotting    = rename_columns(data_market_for_plotting)

ARR_for_plotting            = rename_columns(ARR_for_plotting)

#######################################
## CAR
#######################################

CAR_plot_start  = -25
CAR_plot_end    = 25
CAR_plot_1 = ARR_for_plotting.loc[CAR_plot_start:CAR_plot_end]
CAR_plot_1 = CAR_plot_1.cumsum(axis=0)

CAR_plot_start  = 0
CAR_plot_end    = 25
CAR_plot_2 = ARR_for_plotting.loc[CAR_plot_start:CAR_plot_end]
CAR_plot_2 = CAR_plot_2.cumsum(axis=0)

# Parameters that determine how the plots will look like
fig_width  = 16 # Width of the figure
fig_length = 8 # Length of the figure
linewidth  = 3  # Width of the lines in the plots
fontsize   = 27

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(fig_width,fig_length))
fig.patch.set_facecolor('white')
ax1.set_facecolor('#EEEEEE')
ax2.set_facecolor('#EEEEEE')
ax1.plot(CAR_plot_1,linewidth=linewidth)
ax2.plot(CAR_plot_2,linewidth=linewidth)
ax1.axvline(0,color='orangered',linewidth=linewidth-1,linestyle='--')
ax2.axvline(0,color='orangered',linewidth=linewidth-1,linestyle='--')
ax1.axvline(-3,color='#131627',linewidth=linewidth-1,linestyle='--')
temp = list(CAR_plot_1.columns)
if analyze_countries_or_industries == 'industries':
    temp[-1] = 'Ikke-sykliske\nforbruksvarer'
plt.legend(temp,fontsize=fontsize,loc='upper left',ncol=1,bbox_to_anchor=(1,1))
ax1.grid(True)
ax2.grid(True)
ax1.set_ylabel('CAR',fontsize=fontsize)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax1.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
ax1.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
ax2.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
ax2.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
ax1.set_xlim([np.min(CAR_plot_1.index)-1,np.max(CAR_plot_1.index)+1])
ax2.set_xlim([np.min(CAR_plot_2.index)-1,np.max(CAR_plot_2.index)+1])
plt.savefig(folder_name+'/CAR_plot_'+analyze_countries_or_industries+'.png',dpi=150, bbox_inches='tight')
plt.show() # Show plot in Kernel

#######################################
## Trend plot
#######################################

# Parameters that determine how the plots will look like
fig_width  = 8 # Width of the figure
fig_length = 8 # Length of the figure
linewidth  = 3  # Width of the lines in the plots
fontsize   = 16

months_before_and_after = 2
date_to   = event_date + relativedelta(months=months_before_and_after)
date_from = event_date - relativedelta(months=months_before_and_after)

data_plot = data_market_for_plotting[(data_market_for_plotting.index<=date_to)&(data_market_for_plotting.index>=date_from)]
temp = np.sum(pd.isnull(data_plot),axis=1)
data_plot = data_plot[temp==0]

# Scaling so first trading date is 100
for col in data_plot.columns:
    temp            = data_plot[col].copy()
    temp            = 100*temp/temp.iloc[0]
    data_plot[col] = temp.copy()

fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
fig.patch.set_facecolor('white')
ax.set_facecolor('#EEEEEE')
plt.plot(data_plot,linewidth=linewidth)
plt.axvline(event_date,color='orangered',linewidth=linewidth-1,linestyle='--')
temp = list(data_plot.columns)
if analyze_countries_or_industries == 'industries':
    temp[-1] = 'Ikke-sykliske\nforbruksvarer'
plt.legend(temp,fontsize=fontsize,loc='best',framealpha=1)
ax.set_xlabel('Dato',fontsize=fontsize)
ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
plt.xticks(rotation=45)
plt.savefig(folder_name+'/market_plot_'+analyze_countries_or_industries+'.png',dpi=150, bbox_inches='tight')
plt.show() # Show plot in Kernel
