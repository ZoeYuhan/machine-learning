import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    path ='/Users/yuegeng/Downloads/data.csv'
    data = pd.read_csv(path)    
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print model
    print linreg.coef_
    print linreg.intercept_

    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - y_test) ** 2)    # Mean Squared Error
    rmse = np.sqrt(mse)     # Root Mean Squared Error
    print mse, rmse
    
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
