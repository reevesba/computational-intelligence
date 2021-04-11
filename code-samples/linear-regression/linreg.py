''' Linear Regression Class

Author: Bradley Reeves
Date:   04/10/2021

Code adapted from Chapter 3 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np

class LinearRegressor:
    def __init__(self):
        ''' Initialize LinearRegressor instance
            ----------
            self : object
                LinearRegressor instance
            Returns
            -------
            None
        '''
        self.X = [[]]
        self.beta = [[]]

    def fit(self, X_train, y_train):
        ''' Calculates line of best fit
            ----------
            self : object
                LinearRegressor instance
            X_train : 2d numpy array
                Collection of input vectors x1, x2,...,xn
            y_train : 2d numpy array
                supervised learning labels
            Returns
            -------
            beta : 2d numpy array
                coefficient that defines line of best fit

        '''
        self.X = np.concatenate((X_train, -np.ones((np.shape(X_train)[0], 1))), axis=1)
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)), y_train)
        return self.beta

    def predict(self, X_test):
        ''' Predict target values for test data
            ----------
            self : object
                LinearRegressor instance
            X_test : 2d numpy array
                Collection of input vectors x1, x2,...,xn
            Returns
            -------
            2d numpy array
                Predicted values
        '''
        return np.dot(X_test, self.beta)

    def sse(self, y_pred, y_test):
        ''' Calculate Sum of Squares error
            ----------
            self : object
                LinearRegressor instance
            y_pred : 2d numpy array
                predicted target values
            y_test : 2d numpy array
                actual target values
            Returns
            -------
            float
                total error
        '''
        return np.sum((y_pred - y_test)**2)