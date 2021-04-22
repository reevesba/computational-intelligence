''' MNIST example

Author: Bradley Reeves
Date:   04/21/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''
import pylab as plt
import numpy as np
from mlp import MLP

def main():
    num_samples = 200

    # Load the dataset
    X_train = np.loadtxt('../../datasets/datasets/mnist-train.csv', dtype=int, delimiter=',', max_rows=num_samples)
    X_test = np.loadtxt('../../datasets/datasets/mnist-test.csv', dtype=int, delimiter=',', max_rows=num_samples)

    # Split test set into test and validation sets
    X_valid = X_test[100:, :]
    X_test = X_test[:100, :]

    # 1 of N encoding
    y_train = np.zeros((num_samples, 10))
    for i in range(num_samples):
        y_train[i, X_train[i][0]] = 1

    y_test = np.zeros((int(num_samples/2), 10))
    for i in range(int(num_samples/2)):
        y_test[i, X_test[i][0]] = 1

    y_valid = np.zeros((int(num_samples/2), 10))
    for i in range(int(num_samples/2)):
        y_valid[i, X_valid[i][0]] = 1

    # Experiment with different numbers of hidden nodes
    for i in [1 , 2, 5, 10, 20]:  
        print("----- " + str(i))
        model = MLP(X_train, y_train, num_nodes=i, out_type='softmax')
        model.early_stop(X_train, y_train, X_valid, y_valid, eta=0.1)
        model.confmat(X_test, y_test)

if __name__ == "__main__":
    main()
