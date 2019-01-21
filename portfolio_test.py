# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:00:57 2018

@author: chuny
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web

tickers=['aiz','an','sig','kors','bbby','gt','navi','pvh','rig']

df=web.DataReader(tickers,'yahoo',start='2017-01-01',end='2018-01-01')
data=pd.DataFrame()
for ticker in tickers:
    data[ticker]=df['Adj Close'][ticker]



def plot_close(data):
    
    
    data_columns=data.columns
    for i in data_columns:
#        plt.figure(figsize=(10,6))
        plt.plot(data[i],label='%s Equity Curve ' %(i.upper()))
        plt.legend(loc=0)
        plt.ylabel('Price')
        
#        plt.title('%s Equity Curve ' %(i.upper()))
        #plt.show()
        
plot_close(data)

