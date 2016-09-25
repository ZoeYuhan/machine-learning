import numpy as np
from sklearn.mixture import GMM
from sklearn.cross_validation import train_test_split
import matplotlib as mpl
import matplotlib .colors
import maplotlib.pyplot as plt

def expand(a,b):
    d=(b-a)*0.5
    return a-d,b+d
    
if __name__=='__main__':
    data=np.loadtxt('data.csv',dtype=float)
    print data.shape
    y,x=np.split(data,[1,],axis=1)
    x,x_test,y,y_test=train_test_split(x,y,train_size=0.6,random_state=0)
    gmm=GMM(n_components=2,covariance_type='full',tol=0.0001,n_iter=100,random_state=0)
    x_min,x_max=np.min(x,axis=0,),np.max(x,axis=0)
    gmm.fit(x)
    y_hat=gmm.predict(x_test)
    change=(gmm.means_[0][0]>gmm.means_[1][0])
    if change:
        z=y_hat==0
        y_hat[z]=1
        y_hat[~z]=0
        z=y_test_hat==0
        y_test_hat[z]=1
        y_test_hat[~z]=0
        
     acc = np.mean(y_hat.ravel() == y.ravel())
     acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
     acc_str = 'train correct rate：%.2f%%' % (acc * 100)
     acc_test_str = 'test correct rate：%.2f%%' % (acc_test * 100)
     print acc_str
     print acc_test_str
