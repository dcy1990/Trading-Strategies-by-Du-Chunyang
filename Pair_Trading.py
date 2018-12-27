# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:17:29 2018

@author: chuny
"""
from pandas_datareader import data
from statsmodels.tsa.stattools import coint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

start = '2012-01-01'
end = '2016-12-31'
# Load prices data for HP and Microsoft
df = data.DataReader( ['MSFT','HP'], 'yahoo',start=start, end=end)
X = df['Adj Close'][['MSFT']]

Y = df['Adj Close'][['HP']]
# Compute the p-value for the cointegration of the two series
_, pvalue, _ = coint(X,Y)
print( pvalue)
val=pd.DataFrame()
val['diff']=X['MSFT']-Y['HP']
val['mu']=[val['diff'][:i].mean() for i in range(len(val['diff']))]
mu_60d = val['diff'].rolling(window=60).mean()

fig,ax=plt.subplots(2,1,figsize=(15,7))

ax[1].plot(val['diff'],label='Diff')
ax[1].plot(val['mu'],label='MU')
ax[1].plot(mu_60d,label='60d MU')
ax[0].plot(X,label='MSFT')
ax[0].plot(Y,label='HP')


zscores = [(val['diff'][i] - val['mu'][i]) / 
            np.std(val['diff'][:i]) for i in range(len(val['diff']))]
money = [0]
count = [0]
for i in range(len(val)):
    # Sell short if the z-score is > 1
    if zscores[i] > 1:
        money.append(money[-1]+val['diff'][i])
        count.append(count[-1]-1)
    # Buy long if the z-score is < 1
    elif zscores[i] < -1:
        money.append(money[-1]-val['diff'][i])
        count.append(count[-1]+1)
    # Clear positions if the z-score between -.5 and .5
    elif abs(zscores[i]) < 0.5:
        money.append(money[-1]+count[-1]*val['diff'][i])
        count.append(0)
    else:
        money.append(money[-1])
        count.append(count[-1])
money.pop(0)
count.pop(0)
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,8))
ax1.plot(val.index,money,color='k',label='Cum money')

ax2.plot(val.index,zscores,color='k',label='Z-Score')
ax2.axhline(y=0)
ax2.axhline(y=1)
ax2.axhline(y=-1)
ax3.plot(val.index,count,color='k',label='Position')
ax1.legend()
ax2.legend()
ax3.legend()