#!/usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris

if __name__=='__main__':
    iris = load_iris()
    x, y = iris.data, iris.target

    
#painting
    x_2=x[ : , :2]
    clf_2 = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=3)
    dt_clf_2 = clf_2.fit(x_2, y)
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x_2[:, 0].min(), x_2[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x_2[:, 1].min(), x_2[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    y_hat = dt_clf_2.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.pcolormesh(x1, x2, y_hat, cmap=plt.cm.Spectral, alpha=0.1)  # 预测值的显示Paired/Spectral/coolwarm/summer/spring/OrRd/Oranges
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.prism)  # 样本的显示   
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.show()
    
#model-4 features
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=3)
    dt_clf = clf.fit(x, y)

# Predict
    y_hat = dt_clf.predict(x)
    y = y.reshape(-1)       # 此转置仅仅为了print时能够集中显示
    print y_hat.shape       # 不妨显示下y_hat的形状
    print y.shape
    result = (y_hat == y)   # True则预测正确，False则预测错误
    print y_hat
    print y
    print result
    c = np.count_nonzero(result)    # 统计预测正确的个数
    print c
    print 'Accuracy: %.2f%%' % (100 * float(c) / float(len(result)))
