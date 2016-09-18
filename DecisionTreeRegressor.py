#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

if __name__=='__main__':
    N=100
    x=np.random.rand(N)*6-3
    x.sort()
    x=x.reshape(-1,1)  #N个样本
    depth=[2,4,6,8,10]
    clr='ramby'
    reg=[DecisionTreeRegressor(creterion='mse',max_depth=depth[0]),
        DecisionTreeRegressor(creterion='mse',max_depth=depth[1]),
        DecisionTreeRegressor(creterion='mse',max_depth=depth[2]),
        DecisionTreeRegressor(creterion='mse',max_depth=depth[3]),
        DecisionTreeRegressor(creterion='mse',max_depth=depth[4])]
    plt.plot(x,y,'ko',linewidth=2,label='Actual')
    x_test=np.linspace(-3,3,50).reshape(-1,1)
    for i,r in enumerate(reg):
        dt=r.fit(x,y)
        y_hat=dt.predict(x_test)
        plt.plot(x_test,y_hat,'-',color=clr[i],linewidth=2,label='Depth=%d' %depth[i])
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
