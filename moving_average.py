# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:53:42 2018

@author: chuny
Based on https://www.learndatasci.com/tutorials/python-finance-part-2-intro-quantitative-trading-strategies/

"""

from pandas_datareader import data
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates

my_year_mpnth_fmt=mdates.DateFormatter('%m/%y')


tickers=['AAPL','MSFT','^GSPC']

start_date='20100101'
end_date='20161231'
close = pd.DataFrame()
for ticker in tickers:
    panel_data=data.DataReader(ticker,'yahoo',start_date,end_date)
    close[ticker]=panel_data['Close']

all_weekdays=pd.date_range(start=start_date,end=end_date,freq='B')

close=close.reindex(all_weekdays)
close=close.fillna(method='ffill')

short_rolling=close.rolling(window=20).mean()
long_rolling = close.rolling(window=100).mean()

fig,ax0=plt.subplots(figsize=(10,8))

ax0.plot(close.loc[start_date:end_date,:].index,close.loc[start_date:end_date,'MSFT'],label='MSFT price')
ax0.plot(close.loc[start_date:end_date,:].index,short_rolling.loc[start_date:end_date,'MSFT'],label='MSFT price MA 20')
ax0.plot(close.loc[start_date:end_date,:].index,long_rolling.loc[start_date:end_date,'MSFT'],label='MSFT price MA 100')
ax0.legend()



ema_short = close.ewm(span=20,adjust=False).mean()

trading_positions_raw= close - ema_short
trading_positions=trading_positions_raw.apply(np.sign)*1/3
trading_positions_final=trading_positions.shift(1)

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8))

ax1.plot(close.loc[start_date:end_date,:].index,close.loc[start_date:end_date,'MSFT'],label='MSFT price')
ax1.plot(close.loc[start_date:end_date,:].index,short_rolling.loc[start_date:end_date,'MSFT'],label='MSFT price MA 20')
ax2.plot(close.loc[start_date:end_date,:].index,trading_positions_final.loc[start_date:end_date,'MSFT'],label='MSFT position')
ax1.legend()
ax2.legend()



asset_log_returns=np.log(close).diff()
strategy_log_returns= trading_positions_final*asset_log_returns

cum_strategy_asset_log_returns=strategy_log_returns.cumsum()
cum_strategy_asset_relative_returns=np.exp(cum_strategy_asset_log_returns)-1


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))

for c in asset_log_returns:
    ax1.plot(cum_strategy_asset_log_returns.index, cum_strategy_asset_log_returns[c], label=str(c))

ax1.set_ylabel('Cumulative log-returns')
ax1.legend(loc='best')

for c in asset_log_returns:
    ax2.plot(cum_strategy_asset_relative_returns.index, 100*cum_strategy_asset_relative_returns[c], label=str(c))

ax2.set_ylabel('Total relative returns (%)')
ax2.legend(loc='best')
