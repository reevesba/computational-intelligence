''' Single Layer Perceptron Demo using X Dataset

Author: Bradley Reeves
Date:   04/10/2021

Code adapted from Chapter 3 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import pylab as pl
import numpy as np
from pcn_np import Perceptron

def main():
    eta = 0.25
    iterations = 100

    X = np.loadtxt('../../datasets/diabetes.csv', delimiter=',')

    # Plot the first and second values for the two classes
    indices0 = np.where(X[:, 8] == 0)
    indices1 = np.where(X[:, 8] == 1)

    pl.ion()
    pl.plot(X[indices0, 0], X[indices0, 1], 'go')
    pl.plot(X[indices1, 0], X[indices1, 1], 'rx')

    # Perceptron training on the original dataset
    print("Output on Original Data:")
    model_a = Perceptron(X[:, :8], X[:, 8:9])
    model_a.train(eta, iterations)
    model_a.confusion_matrix(X[:, :8], X[:, 8:9])

    # Various preprocessing steps
    # If pregnant more than 8 times, set value to 8
    X[np.where(X[:, 0] > 8), 0] = 8

    # Quantise ages into set of ranges
    # Group 1: 30 and below
    # Group 2: 31 - 40
    # Group 3: 41 - 50
    # Group 4: 51 - 60
    # Group 5: 61 and above
    X[np.where(X[:, 7] <= 30), 7] = 1
    X[np.where((X[:, 7] > 30) & (X[:, 7] <= 40)), 7] = 2
    X[np.where((X[:, 7] > 40) & (X[:, 7] <= 50)), 7] = 3
    X[np.where((X[:, 7] > 50) & (X[:, 7] <= 60)), 7] = 4
    X[np.where(X[:, 7] > 60), 7] = 5

    # Normalizing dataset
    X[:, :8] = X[:, :8] - X[:, :8].mean(axis=0)
    X[:, :8] = X[:, :8]/X[:, :8].var(axis=0)

    X_train = X[::2, :8]
    X_test = X[1::2, :8]
    y_train = X[::2, 8:9]
    y_test = X[1::2, 8:9]

    # Perceptron training on the preprocessed dataset
    print("Output after preprocessing of data")
    model_b = Perceptron(X_train, y_train)
    model_b.train(eta, iterations)
    model_b.confusion_matrix(X_test, y_test)

    pl.savefig('output/pima.png')

if __name__ == "__main__":
    main()