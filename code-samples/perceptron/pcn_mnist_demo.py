''' Single Layer Perceptron Demo using MNIST Dataset

Author: Bradley Reeves
Date:   04/10/2021

Code adapted from Chapter 3 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from pcn_np import Perceptron

def main():
    num_samples = 200
    eta = 0.25
    iterations = 100

    X_train = np.loadtxt('../../datasets/mnist-train.csv', dtype=int, delimiter=',', max_rows=num_samples)
    X_test = np.loadtxt('../../datasets/mnist-test.csv', dtype=int, delimiter=',', max_rows=num_samples)

    # 1 of N encoding
    y_train = np.zeros((num_samples, 10))
    y_test = np.zeros((num_samples, 10))

    for i in range(num_samples):
        y_train[i, X_train[i][0]] = 1
        y_test[i, X_test[i][0]] = 1

    # Train a Perceptron on training set
    model = Perceptron(X_train, y_train)
    model.train(eta, iterations)

    # This isn't really good practice since it's on the training data, 
    # but it does show that it is learning.
    model.confusion_matrix(X_train, y_train)

    # Now test it
    model.confusion_matrix(X_test, y_test)

if __name__ == '__main__':
    main()