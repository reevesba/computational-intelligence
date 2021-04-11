''' Single Layer Perceptron that Prints Weights

Author: Bradley Reeves
Date:   04/10/2021

Code adapted from Chapter 3 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np

class Perceptron:
    def __init__(self, inputs, targets):
        ''' Initialize Perceptron instance
            ----------
            self : object
                Perceptron instance
            inputs : 2d numpy array
                Collection of input vectors x1, x2,...,xn
            targets : 2d numpy array
                supervised learning labels
            Returns
            -------
            None
        '''
        self.X = inputs
        self.y = targets

        if np.ndim(self.X) > 1:
            self.X_dims = np.shape(self.X)[1]
        else:
            self.X_dims = 1

        if np.ndim(self.y) > 1:
            self.y_dims = np.shape(self.y)[1]
        else:
            self.y_dims = 1

        self.num_samples = np.shape(self.X)[0]
        self.weights = np.random.rand(self.X_dims + 1, self.y_dims)*0.1 - 0.05
        self.activations = self.feed_forward()

    def input_with_bias(self):
        ''' Adds bias node to input vectors
            Parameters
            ----------
            self : object
                Perceptron instance
            Returns
            -------
            2d numpy array
                input including bias node
        '''
        return np.concatenate((self.X, -np.ones((self.num_samples, 1))), axis=1)

    def feed_forward(self):
        ''' Computes new activations, acts as thresholding function
            Parameters
            ----------
            self : object
                Perceptron instance
            Returns
            -------
            2d numpy array
                Activations after thresholding
        '''
        return np.where(np.dot(self.input_with_bias(), self.weights) > 0, 1, 0)

    def train(self, eta, iterations):
        ''' Training perceptron
            Parameters
            ----------
            self : object
                Perceptron instance
            eta : float
                Learning rate
            iterations : integer
                Number of training steps
            Returns
            -------
            None
        '''
        for i in range(iterations):
            self.weights -= eta*np.dot(np.transpose(self.input_with_bias()), self.activations - self.y)
            self.activations = self.feed_forward()
            
            print("Iteration: ", i)
            print(self.weights)
            print("Final outputs are:")
            print(self.activations)

    def confusion_matrix(self, inputs, targets):
        ''' Prints confusion matrix to terminal
            Parameters
            ----------
            self : object
                Perceptron instance
            inputs : 2d numpy array
                Collection of input vectors x1, x2,...,xn
            targets : 2d numpy array
                supervised learning labels
            Returns
            -------
            None
        '''
        inputs = np.concatenate((inputs, -np.ones((self.num_samples, 1))), axis=1)
        outputs = np.dot(inputs, self.weights)
        num_classes = np.shape(targets)[1]

        if num_classes==1:
            num_classes = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)*np.where(targets == j, 1, 0))

        print(matrix)
        print(np.trace(matrix)/np.sum(matrix))

