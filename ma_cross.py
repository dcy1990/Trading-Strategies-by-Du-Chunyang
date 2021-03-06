# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:34:47 2019
based on quantStart
https://www.quantstart.com/articles/Research-Backtesting-Environments-in-Python-with-pandas
@author: chuny
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data
from backtest import Strategy, Portfolio


class MarketOnClosePortfolio(Portfolio):

    def __init__(self, symbol, bars, signals, initial_capital=100000.0):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions[self.symbol] = 100*signals['signal']   # This strategy buys 100 shares(stupid assumption)
        return positions
                    
    def backtest_portfolio(self):
#        portfolio = self.positions[self.symbol]*self.bars['Close']
        portfolio=pd.DataFrame()
        portfolio['holdings'] = (self.positions[self.symbol]*self.bars['Close'])
        portfolio['cash'] = self.initial_capital -(self.positions[self.symbol].diff()
            *self.bars['Close']).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio


class MovingAverageCrossStrategy(Strategy):

    def __init__(self, symbol, bars, short_window=100, long_window=400):
        self.symbol = symbol
        self.bars = bars
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        signals['short_mavg'] =bars['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = bars['Close'].rolling(window=self.long_window, min_periods=1).mean()  
        signals['signal']=((signals['short_mavg']-signals['long_mavg']).apply(np.sign)+1)/2
        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()   

        return signals

symbol = 'MSFT'
bars = data.DataReader(symbol, "google", datetime.datetime(2001,1,1), datetime.datetime(2015,1,1))

mac = MovingAverageCrossStrategy(symbol, bars, short_window=100, long_window=400)
signals = mac.generate_signals()

# Create a portfolio of AAPL, with $100,000 initial capital
portfolio = MarketOnClosePortfolio(symbol, bars, signals, initial_capital=10.0)

returns = portfolio.backtest_portfolio()

# Plot two charts to assess trades and equity curve
fig = plt.figure()
fig.patch.set_facecolor('white')     # Set the outer colour to white
ax1 = fig.add_subplot(211,  ylabel='Price in $')

# Plot the AAPL closing price overlaid with the moving averages
bars['Close'].plot(ax=ax1, color='r', lw=2.)
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the "buy" trades against AAPL
ax1.plot(signals[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')

# Plot the "sell" trades against AAPL
ax1.plot(signals[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')

# Plot the equity curve in dollars
ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
returns['total'].plot(ax=ax2, lw=2.)

# Plot the "buy" and "sell" trades against the equity curve
ax2.plot(returns[signals.positions == 1.0].index, 
         returns.total[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax2.plot(returns[signals.positions == -1.0].index, 
         returns.total[signals.positions == -1.0],
         'v', markersize=10, color='k')

# Plot the figure
fig.show()