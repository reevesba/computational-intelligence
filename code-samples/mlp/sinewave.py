''' Sinewave regression example

Author: Bradley Reeves
Date:   04/21/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import pylab as plt
import numpy as np
from mlp import MLP

def main():
    # Set up the data
    X = np.linspace(0, 1, 40).reshape((40, 1))
    y = np.sin(2*np.pi*X) + np.cos(4*np.pi*X) + np.random.randn(40).reshape((40, 1))*0.2
    X = (X - 0.5)*2

    # Plot the data
    plt.plot(X, y, 'o')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig('out/sinewave.png')

    # Split into training, testing, and validation sets
    X_train, y_train = X[0::2, :], y[0::2, :]
    X_test, y_test = X[1::4, :], y[1::4, :]
    X_valid, y_valid = X[3::4, :], y[3::4, :]

    # Test out different sizes of network
    count = 0
    out = np.zeros((10, 7))
    for nnodes in [1, 2, 3, 5, 10, 25, 50]:
        for i in range(10):
            model = MLP(X_train, y_train, nnodes, out_type='linear')
            out[i, count] = model.early_stop(X_train, y_train, X_valid, y_valid, 0.25)
        count += 1
        
    X_test = np.concatenate((X_test, -np.ones((np.shape(X_test)[0], 1))), axis=1)
    outputs = model.feed_forward(X_test)
    print("Test Error:", 0.5*sum((outputs - y_test)**2))

    #print("Output:", out)
    print("Mean:", out.mean(axis=0))
    print("Std. Dev.:", out.var(axis=0))
    print("Max Error:", out.max(axis=0))
    print("Min Error:", out.min(axis=0))

    ''' Output Sample
         ____________________________________________________________________________
        | No. of Hidden Nodes | 1       2       3       5       10      25      50   |
        | ---------------------------------------------------------------------------|
        | Mean Error          | 2.61    0.79    0.86    0.86    0.75    0.65    0.66 |
        | Standard Deviation  | 0.00    0.00    0.00    0.00    0.00    0.00    0.00 |
        | Max Error           | 2.61    0.82    0.91    0.93    0.88    0.67    0.74 |
        | Min Error           | 2.61    0.75    0.79    0.71    0.63    0.61    0.63 |
        |____________________________________________________________________________|

        According to this output, 25 hidden nodes may be best.
    '''

if __name__ == "__main__":
    main()
