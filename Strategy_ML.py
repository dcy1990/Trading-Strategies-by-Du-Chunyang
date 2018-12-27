# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 11:36:15 2018

@author: chuny
"""

import numpy as np
import pandas as pd
from pandas_datareader import data
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.metrics import mean_squared_error,mean_absolute_error

num_folds=10
seed=7
scoring = 'neg_mean_squared_error'

df=data.DataReader('SPY','yahoo',start='20100101',end='20170301')
df=df.drop(['Volume','Adj Close'],axis=1)
#df.replace([np.inf, -np.inf], np.nan)
#df=df.dropna(axis=1)
df['open']=df['Open'].shift(1)
df['close']=df['Close'].shift(1)
df['high']=df['High'].shift(1)
df['low']=df['Low'].shift(1)
#df=df.dropna()

#pd.plotting.scatter_matrix(df)

X=df[['open','high','low','close']]
Y=df['Close']

#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=seed)

imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
pipeline=Pipeline([('imputation',imp),
                   ('scaler',StandardScaler()),
                   ('lasso',Lasso())])

#kfold=KFold(n_splits=num_folds,random_state=seed)
#cv_results=cross_val_score(pipeline,X_train,Y_train,cv=kfold,scoring=scoring)
#
#print(cv_results.mean(),cv_results.std())

parameters={'lasso__alpha':np.arange(0.0001,10,0.0001),
            'lasso__max_iter':np.random.uniform(100,100000,10)}
reg=rcv(pipeline,parameters,cv=5)
avg_err={}

for t in np.arange(50,97,3):
    split=int(t*len(X)/100)
    reg.fit(X[:split],Y[:split])
    best_alpha=reg.best_params_['lasso__alpha']
    best_iter=reg.best_params_['lasso__max_iter']
    reg1=Lasso(alpha=best_alpha,max_iter=best_iter)
    X=imp.fit_transform(X,Y)
    result=reg1.fit(X[:split],Y[:split])
    prediction=result.predict(X[split:])
    avg_err[t]=mean_absolute_error(prediction,Y[split:])
    print(t,mean_absolute_error(prediction,Y[split:]))

fig,ax=plt.subplots(figsize=(10,8))
ax.plot(avg_err.keys(),avg_err.values())



'''
scaler=StandardScaler().fit(X_train)
rescaledX=scaler.fit_transform(X_train)

k_values=np.arange(1,22,2)
param_grid=dict(alpha=k_values.tolist())
model=Lasso()
kfold=KFold(n_splits=num_folds,random_state=seed)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_results=grid.fit(rescaledX,Y_train)

print(grid_results.best_score_,grid_results.best_params_)
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''