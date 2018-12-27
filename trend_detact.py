# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 11:36:15 2018
based on https://www.quantinsti.com/blog/trading-using-machine-learning-python-part-2

@author: chuny
"""

import numpy as np
import pandas as pd
from pandas_datareader import data
import talib as ta


df=data.DataReader('SPY','yahoo',start='20100101',end='20170301')
df=df.drop(['Volume','Adj Close'],axis=1)
n=10
t=0.8
spilt=int(t*len(df))


df['open']=df['Open'].shift(1)
df['close']=df['Close'].shift(1)
df['high']=df['High'].shift(1)
df['low']=df['Low'].shift(1)

df['RSI']=ta.RSI(np.array(df['close']),timeperiod=n)
df['SMA']=df['close'].rolling(window=n).mean()
df['Corr']=df['SMA'].rolling(window=n).corr(df['close'])
df['SAR']=ta.SAR(np.array(df['high']),np.array(df['low']),0.2,0.2)
df['ADX']=ta.ADX(np.array(df['high']),np.array(df['low']),
  np.array(df['close']),timeperiod=n)
df['Return']=np.log(df['Open']).diff()

df=df.dropna()
