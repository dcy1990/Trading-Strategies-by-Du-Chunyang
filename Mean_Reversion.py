# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 09:30:29 2018
based on https://gist.github.com/chandinijain/1defc51737f84ba35b11694a78d1a131#file-meanreversion-ipynb

@author: chuny
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

start = '2013-06-01'
end = '2016-12-31'

df=data.DataReader('PG','yahoo',start=start,end=end)
price=df[['Adj Close']]
xx=df['Adj Close'].cumsum()/np.arange(1,len(df)+1)

price = price.assign(mu=xx.values)

fig,ax0=plt.subplots(1,1,figsize=(10,8))
ax0.plot(price.index,price['Adj Close'],color='k',label='PG price')
ax0.plot(price.index,price['mu'],color='r',label='PG CumSum average price')
ax0.legend()


zscores = [(price['Adj Close'][i] - price['mu'][i])/
            np.std(price['Adj Close'][:i]) for i in range(len(price))]

money = [0]
count = [0]
for i in range(len(price)):
    # Sell short if the z-score is > 1
    if zscores[i] > 1:
        money.append(money[-1]+price['Adj Close'][i])
        count.append(count[-1]-1)
    # Buy long if the z-score is < 1
    elif zscores[i] < -1:
        money.append(money[-1]-price['Adj Close'][i])
        count.append(count[-1]+1)
    # Clear positions if the z-score between -.5 and .5
    elif abs(zscores[i]) < 0.5:
        money.append(money[-1]+count[-1]*price['Adj Close'][i])
        count.append(0)
    else:
        money.append(money[-1])
        count.append(count[-1])
money.pop(0)
count.pop(0)
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,8))
ax1.plot(price.index,money,color='k',label='Cum money')

ax2.plot(price.index,zscores,color='k',label='Z-Score')
ax2.axhline(y=0)
ax2.axhline(y=1)
ax2.axhline(y=-1)
ax3.plot(price.index,count,color='k',label='Position')
ax1.legend()
ax2.legend()
ax3.legend()
'''






start = '2016-12-01'
end = '2016-12-31'
assets = ['AAPL', 'AIG', 'C', 'T', 'PG', 'JNJ', 'EOG', 'MET', 'AMGN']

prices=pd.DataFrame()
for i in assets:
    df=data.DataReader(i,'yahoo',start=start,end=end)
    prices[i]=df['Adj Close']

returns=prices.pct_change(1)
colors=['r', 'g', 'b', 'k', 'c', 'm', 'orange','chartreuse', 'slateblue']
returns.plot(figsize=(10,8),color=colors)
plt.legend()
'''

#
#returns = prices/prices.shift(-1) -1
#returns.plot(figsize=(15,7), color=['r', 'g', 'b', 'k', 'c', 'm', 'orange',
#                                     'chartreuse', 'slateblue', 'silver'])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.ylabel('Returns')
#
## Convert to numpy array to make manipulation easier
#data = np.array(prices);
