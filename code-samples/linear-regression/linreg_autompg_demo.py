''' Linear Regression Demo on Auto MPG Dataset

Author: Bradley Reeves
Date:   04/10/2021
'''

import numpy as np
from linreg import LinearRegressor

def main():
    data = np.loadtxt('../../datasets/auto-mpg.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    np.random.shuffle(data)

    # 80/20 test/train split
    split = int(len(data)*.8)

    # Normalize the data (doesn't seem to make a difference)
    #data[:, 1:] = data[:, 1:] - data[:, 1:].mean(axis=0)
    #data[:, 1:] = data[:, 1:]/data[:, 1:].var(axis=0)
    
    train, test = np.split(data, [split])
    X_train = train[:, 1:]
    X_test = test[:, 1:]
    y_train = train[:, :1]
    y_test = test[:, :1]

    # Supplement test data with bias node
    X_test = np.concatenate((X_test, -np.ones((np.shape(X_test)[0], 1))), axis=1)

    model = LinearRegressor()
    beta = model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = model.sse(pred, y_test)

    print("Auto MPG Linear Regression Demo")
    print("beta:\n", beta)
    print("predictions:\n", pred)
    print("error:\n", error)

if __name__ == "__main__":
    main()