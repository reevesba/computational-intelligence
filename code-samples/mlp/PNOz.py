''' Palmerston North Ozone time series example

Author: Bradley Reeves
Date:   04/21/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import pylab as plt
import numpy as np
from mlp import MLP

def main():
    dataset = np.loadtxt('../../datasets/datasets/palmerson-north-ozone.dat')

    # Plot the dataset
    plt.ion()
    plt.plot(np.arange(np.shape(dataset)[0]), dataset[:, 2], '.')
    plt.xlabel('Time (Days)')
    plt.ylabel('Ozone (Dobson units)')
    plt.savefig('out/ozone.png')

    # Normalise dataset
    dataset[:, 2] = dataset[:, 2] - dataset[:, 2].mean()
    dataset[:, 2] = dataset[:, 2]/dataset[:, 2].max()

    # Assemble input vectors
    t, k = 2, 3

    last_pt = np.shape(dataset)[0] - t*(k + 1)
    X = np.zeros((last_pt, k))
    y = np.zeros((last_pt, 1))

    for i in range(last_pt):
        X[i, :] = dataset[i:i + t*k:t, 2]
        y[i] = dataset[i + t*(k + 1), 2]
        
    X_train, y_train = X[:-400:2, :], y[:-400:2]
    X_valid, y_valid = X[1:-400:2, :], y[1:-400:2]
    X_test, y_test = X[-400:, :], y[-400:]

    # Randomly order the data
    change = np.arange(np.shape(X)[0])
    np.random.shuffle(change)
    X = X[change, :]
    y = y[change, :]

    # Train the network
    model = MLP(X_train, y_train, num_nodes=3, out_type='linear')
    model.early_stop(X_train, y_train, X_valid, y_valid, eta=0.25)

    test = np.concatenate((X_test, -np.ones((np.shape(X_test)[0], 1))), axis=1)
    testout = model.feed_forward(test)

    # Plot the results
    plt.figure()
    plt.plot(np.arange(np.shape(test)[0]), testout, '.')
    plt.plot(np.arange(np.shape(test)[0]), y_test, 'x')
    plt.legend(('Predictions', 'Targets'))
    #print(0.5*np.sum((y_test - testout)**2))
    plt.savefig('out/time-series-results.png')

if __name__ == "__main__":
    main()
