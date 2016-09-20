#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

if __name__=='__main__':
    iris=load_iris()
    x,y=iris.data,iris.target
    
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(12,10).facecolor='#FFFFFF')
    
    for i,pair in enumerate(feature_pairs):
      x=x[:,pair]
      
      clf=RandomForestClassifier(n_estimators=50,criterion='entropy')
      rf_clf=clf.fit(x,y)
      
#painting
    N,M=500,500
    x1_min,x1_max=x[:,0].min(), x[:,0].max()  #0th max min
    x2_min,x2_max=x[:,1].min(), x[:,0].max()  #1th max min
    t1=np.linespace(x1_min,x1_max,N)
    t2=np.linespace(x2_min,x2_max,M)
    x1,x2=np.meshgrid(t1,t2)
    x_test=np.stack((x1.flat,x2.flat),axis=1)
    
#Predict
    y_hat=rf_clf.predict(x)
    y=y.reshape(-1)
    right=np.count_nonzoro(y_hat==y)
    print 'right rate is: %.2f%%' % (100*float(right)/float(len(y)))
    
#show
    y_hat=rf.clf_predict(x_test)
    y_hat=y_hat.reshape(x1.shape)
    plt.subplot(2,3,i+1)
    plt.pcolormesh(x1,x2,y_hat,cmap=plt.cm.Spectral,alpha=0.5)
    plt.scatter(x[:,0],x[:,1],right=y,edgecolors='k',cmap=plt.cm.prism)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.grid()
    plt.axis('tight')
    
plt.show()
