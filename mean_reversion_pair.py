# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:38:13 2019
based on https://www.quantstart.com/articles/Backtesting-An-Intraday-Mean-Reversion-Pairs-Strategy-Between-SPY-And-IWM

@author: chuny
"""

from pandas_datareader import data
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint



def create_pairs_dataframe(time_interval, symbols):


    df=data.DataReader(symbols,'yahoo',time_interval[0],time_interval[1])
    pairs = pd.DataFrame(index=df.index)
    pairs['%s_close' % symbols[0].lower()] = df['Close'][symbols[0]]
    pairs['%s_close' % symbols[1].lower()] = df['Close'][symbols[1]]
    pairs = pairs.dropna()
    return pairs

def calculate_spread_zscore(pairs, symbols, lookback=100):

#    model = coint(y=pairs['%s_close' % symbols[0].lower()], 
#                   x=pairs['%s_close' % symbols[1].lower()],
    _, pvalue, _ = coint(pairs['%s_close' % symbols[1].lower()],
                        pairs['%s_close' % symbols[0].lower()])
    print( pvalue)
    pairs['hedge_ratio'] = pvalue
    pairs = pairs.dropna()
    pairs['spread'] = pairs['spy_close'] - pairs['hedge_ratio']*pairs['iwm_close']
    pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread']))/np.std(pairs['spread'])
    return pairs


def create_long_short_market_signals(pairs, symbols, 
                                     z_entry_threshold=2.0, 
                                     z_exit_threshold=1.0):

    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

    pairs['long_market'] = 0.0
    pairs['short_market'] = 0.0

    long_market = 0
    short_market = 0

    for i, b in enumerate(pairs.iterrows()):
        # Calculate longs
        if b[1]['longs'] == 1.0:
            long_market = 1            
        # Calculate shorts
        if b[1]['shorts'] == 1.0:
            short_market = 1
        # Calculate exists
        if b[1]['exits'] == 1.0:
            long_market = 0
            short_market = 0
        # This directly assigns a 1 or 0 to the long_market/short_market
        # columns, such that the strategy knows when to actually stay in!
        pairs.ix[i]['long_market'] = long_market
        pairs.ix[i]['short_market'] = short_market
    return pairs


def create_portfolio_returns(pairs, symbols):
    sym1 = symbols[0].lower()
    sym2 = symbols[1].lower()

    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['long_market'] - pairs['short_market']
    portfolio[sym1] = -1.0 * pairs['%s_close' % sym1] * portfolio['positions']
    portfolio[sym2] = pairs['%s_close' % sym2] * portfolio['positions']
    portfolio['total'] = portfolio[sym1] + portfolio[sym2]

    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    # Calculate the full equity curve
    portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
    return portfolio

symbols = ('SPY', 'IWM')

time_intreval=['20070401','20140701']
returns = []

lookbacks = range(50, 210, 10)
returns = []

# Adjust lookback period from 50 to 200 in increments
# of 10 in order to produce sensitivities
for lb in lookbacks: 
    print ("Calculating lookback=%s..." % lb)
    pairs = create_pairs_dataframe(time_intreval, symbols)
    pairs = calculate_spread_zscore(pairs, symbols, lookback=lb)
    pairs = create_long_short_market_signals(pairs, symbols, 
                                            z_entry_threshold=2.0, 
                                            z_exit_threshold=1.0)

    portfolio = create_portfolio_returns(pairs, symbols)
    returns.append(portfolio.ix[-1]['returns'])

plt.plot(portfolio.index, portfolio.returns, '-')
plt.show()