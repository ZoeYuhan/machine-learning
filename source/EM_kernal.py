#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GMM
import matplotlib as mpl
improt matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__=='__main__':
    miu1=(0,0,0)
    Deta1=np.identity(3)
    data1=np.random.multivariate_normal(miu1,Deta1,400)
    
    miu2=(2,2,1)
    Deta2=np.identity(3)
    data2=np.random.multivariate_normal(miu2,Deta2,100)
    
    data=np.vstack((data1,data2))
    
    num=100
    row,col=data.shape
    
    #initial miu ---Random
    miu1=np.random.standard_normal(col)
    miu2=np.random.standard_normal(col)
    
    sigma1=np.identity(col)
    sigma2=np.identity(col)
    
    rate=0.5
    
    for i in range(num):
    #E-step
      norm1=multivariate_normal(miu1,sigma1)
      norm2=multivatiate_normal(miu2,sigma2)
      tau1=rate*norm1.pdf(data)
      tau2=(1-rate)*norm2.pdf(data)
      gamma=tau1/(tau1+tau2)
      
    #M-step
      miu1=np.dot(gamma,data)/sum(gamma)
      miu2=np.dot((1-gamma),data)/sum((1-gamma))
      sigma1=np.dot(gamma*(data-miu1).T,data-miu1)/np.sum(gamma)
      sigma2=np.dot((1-gamma)*(data-miu2).T,data-miu2)/np.sum((1-gamma))
      rate=sum(gamma)/n
    print '概率类别:\t',rate
    print '均值:\t ', miu1,miu2
    print '方差:\t ', sigma1,sigma2 
      
    g=GMM(n_components=2,covariance_type='full',n_iter=100)
    g.fit(data)
    print '-------GMM------'
    print '概率类别:\t',g.weights_[0]
    print '均值:\t ', g.means_,'\n'
    print '方差:\t ', g.covars_,'\n' 
    
    #predict
    norm1=multivariate_normal(miu1,sigma1)
    norm2=multivariate_normal(miu2,sigma2)
    tau1=norm1.pdf(data)
    tau2=norm2.pdf(data)
    
    tau=g.predict(data)
    
    fig = plt.figure(figsize=(14, 7), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'原始数据', fontsize=18)
    ax = fig.add_subplot(122, projection='3d')
    c1 = tau1 > tau2
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    c2 = tau1 < tau2
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM算法分类', fontsize=18)

    plt.tight_layout()
    plt.show()
    
      
      
  
      


