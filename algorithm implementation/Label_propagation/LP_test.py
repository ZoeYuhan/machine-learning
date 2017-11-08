#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/26 9:46
# @Author  : Zoe
# @Site    : 
# @File    : LP_test.py
# @Software: PyCharm Community Edition

import time
import math
import numpy as np
from LP import labelPropagation


# show
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels,step):
    import matplotlib.pyplot as plt

    for i in range(Mat_Label.shape[0]):
        if int(labels[i]) == 0:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'ro',alpha=0.5)
        elif int(labels[i]) == 1:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'b*',alpha=0.5)
        else:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'y*',alpha=0.5)

    for i in range(Mat_Unlabel.shape[0]):
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'r^',alpha=0.5)
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'b.',alpha=0.5)
        else:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'y.',alpha=0.5)

    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.xlim(-2.0, 15.)
    plt.ylim(-2.0, 15.)

    plt.title("step=%d"%step)
    plt.legend()
    plt.show()


def loadCircleData(num_data):
    center = np.array([6.0, 6.0])
    radiu=180
    radiu_inner = 3
    radiu_outer = 6
    num_inner = num_data / 3
    num_outer = num_data - num_inner

    data1 = []
    data2 = []
    Mat_Label = np.zeros((2, 2), np.float32)
    theta = 0.0
    for i in range(int(num_inner)):
        pho = 180+(theta % radiu) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data1.append(tmp)
        theta += 2
    Mat_Label[0] =np.array(tmp)
    theta = 0.0
    for i in range(int(num_outer)):
        pho = (theta % (radiu)) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data2.append(tmp)
        theta += 1
    Mat_Label[1] = np.array(tmp)

    # Mat_Label[0] = center + np.array([-radiu_inner + 0.5, 0])
    # Mat_Label[1] = center + np.array([-radiu_outer + 0.5, 0])

    labels = [0, 1]
    Mat_Unlabel = np.vstack((data1,data2))
    return Mat_Label, labels, Mat_Unlabel


def loadBandData(num_unlabel_samples):
    # Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    # labels = [0, 1]
    # Mat_Unlabel = np.array([[5.1, 2.], [5.0, 8.1]])

    Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    labels = [0, 1]
    num_dim = Mat_Label.shape[1]
    Mat_Unlabel = np.zeros((num_unlabel_samples, num_dim), np.float32)
    Mat_Unlabel[:num_unlabel_samples / 2, :] = (np.random.rand(num_unlabel_samples / 2, num_dim) - 0.5) * np.array(
        [3, 1]) + Mat_Label[0]
    Mat_Unlabel[num_unlabel_samples / 2: num_unlabel_samples, :] = (np.random.rand(num_unlabel_samples / 2,
                                                                                   num_dim) - 0.5) * np.array([3, 1]) + \
                                                                   Mat_Label[1]
    return Mat_Label, labels, Mat_Unlabel


# main function
if __name__ == "__main__":
    num_unlabel_samples = 1000
    # Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)
    Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)
    unlabel_data_labels=np.array([2]*Mat_Unlabel.shape[0])
    show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels,0)
   
    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    # unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)
    for i in [1,10,30,100,150,300,500,1000]:
        unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='knn', knn_num_neighbors=10,
                                               max_iter=i)
        show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels , i)
