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
import seaborn as sns
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

returns = close.pct_change(1)
log_returns = np.log(close).diff()
'''
fig,ax0=plt.subplots(1,1,figsize=(10,8))

ax0.plot(close.index,(np.exp(log_returns['AAPL'].cumsum()) - 1),'k',label='Returns')
ax0.plot(close.index,log_returns['AAPL'].cumsum(),'r',label='Log Returns')
ax0.legend()

'''
weights_matrix= pd.DataFrame(1/3, index=close.index, columns=close.columns)
temp_var = weights_matrix.dot(log_returns.transpose())
portfolio_log_returns = pd.Series(np.diag(temp_var), index=log_returns.index)

total_relative_returns = (np.exp(portfolio_log_returns.cumsum()) - 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))

ax1.plot(portfolio_log_returns.index, portfolio_log_returns.cumsum(),label='Cumulative log return')
ax1.set_ylabel('Portfolio cumulative log returns')
ax1.legend()
ax2.plot(total_relative_returns.index, 100 * total_relative_returns,label='Portfolio total relative returns')
ax2.set_ylabel('Portfolio total relative returns (%)')
ax2.legend()
plt.show()
