import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
x, y = iris.data, iris.target

clf = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=3)
dt_clf = clf.fit(x, y)

 # 训练集上的预测结果
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
