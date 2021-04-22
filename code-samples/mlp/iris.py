''' Iris Classification Example

Author: Bradley Reeves
Date:   04/21/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from mlp import MLP

def preprocessIris(infile, outfile):
    # Current class values
    class1 = 'Iris-setosa'
    class2 = 'Iris-versicolor'
    class3 = 'Iris-virginica'

    # Replacement class values
    new_class1 = '0'
    new_class2 = '1'
    new_class3 = '2'

    fid = open(infile,"r")
    oid = open(outfile,"w")

    for s in fid:
        if s.find(class1)>-1:
            oid.write(s.replace(class1, new_class1))
        elif s.find(class2)>-1:
            oid.write(s.replace(class2, new_class2))
        elif s.find(class3)>-1:
            oid.write(s.replace(class3, new_class3))

    fid.close()
    oid.close()

def main():
    # Preprocessor to remove the test (only needed once)
    #preprocessIris('../../datasets/iris.csv', 'iris_proc.data')

    # Load and normalize the dataset
    iris = np.loadtxt('iris_proc.data', delimiter=',')
    iris[:, :4] = iris[:, :4] - iris[:, :4].mean(axis=0)
    imax = np.concatenate((iris.max(axis=0)*np.ones((1, 5)), np.abs(iris.min(axis=0)*np.ones((1, 5)))), axis=0).max(axis=0)
    iris[:, :4] = iris[:, :4]/imax[:4]
    #print(iris[0:5, :])

    # 1-of-N encoding
    target = np.zeros((np.shape(iris)[0], 3))
    indices = np.where(iris[:, 4] == 0) 
    target[indices, 0] = 1
    indices = np.where(iris[:, 4] == 1)
    target[indices, 1] = 1
    indices = np.where(iris[:, 4] == 2)
    target[indices, 2] = 1

    # Randomly order the data
    order = np.arange(np.shape(iris)[0])
    np.random.shuffle(order)

    # Split into train, validation, and test sets
    iris = iris[order, :]
    target = target[order, :]

    X_train, y_train = iris[::2, 0:4], target[::2]
    X_valid, y_valid = iris[1::4, 0:4], target[1::4]
    X_test, y_test = iris[3::4, 0:4], target[3::4]
    #print train.max(axis=0), train.min(axis=0)

    # Train the network
    model = MLP(X_train, y_train, num_nodes=5, out_type='logistic')
    model.early_stop(X_train, y_train, X_valid, y_valid, eta=0.1)
    model.confmat(X_test, y_test)

if __name__ == "__main__":
    main()
