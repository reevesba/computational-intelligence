''' Linear Regression Demo on Basic Logic Functions

Author: Bradley Reeves
Date:   04/10/2021

Code adapted from Chapter 3 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from linreg import LinearRegressor

def main():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    X_test = np.concatenate((X_train, -np.ones((np.shape(X_train)[0], 1))), axis=1)

    # Targets
    and_y = np.array([[0], [0], [0], [1]])
    or_y = np.array([[0], [1], [1], [1]])
    xor_y = np.array([[0], [1], [1], [0]])

    print("Testing AND Logic Function:")
    and_model = LinearRegressor()
    and_beta = and_model.fit(X_train, and_y)
    and_out = and_model.predict(X_test)
    and_out = np.where(and_out < 0.5, 0, 1)
    print(and_out)

    print("Testing OR Logic Function:")
    or_model = LinearRegressor()
    or_beta = or_model.fit(X_train, or_y)
    or_out = or_model.predict(X_test)
    or_out = np.where(or_out < 0.5, 0, 1)
    print(or_out)

    print("Testing XOR Logic Function:")
    xor_model = LinearRegressor()
    xor_beta = xor_model.fit(X_train, xor_y)
    xor_out = xor_model.predict(X_test)
    xor_out = np.where(xor_out < 0.5, 0, 1)
    print(xor_out)

if __name__ == '__main__':
    main()