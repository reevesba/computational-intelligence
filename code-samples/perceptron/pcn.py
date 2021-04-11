''' Single Layer Perceptron in Vanilla Python

Author: Bradley Reeves
Date:   04/10/2021

Code adapted from Chapter 3 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from random import seed, random

class Perceptron:
    def __init__(self, inputs, targets):
        ''' Initialize Perceptron instance
            ----------
            self : object
                Perceptron instance
            inputs : 2d list
                Collection of input vectors x1, x2,...,xn
            targets : 2d list
                supervised learning labels
            Returns
            -------
            None
        '''
        self.X = inputs
        self.y = targets

        if len(self.X[0]) > 1:
            self.input_dims = len(self.X[0])
        else:
            self.input_dims = 1

        if len(self.y[0]) > 1:
            self.target_dims = len(self.y[0])
        else:
            self.target_dims = 1

        self.num_vectors = len(self.X)
        self.weights = [[random()*0.1 - 0.05 for i in range(self.target_dims)] for j in range(self.input_dims + 1)]
        self.activations = self.feed_forward()

    def input_with_bias(self):
        ''' Adds bias node to input vectors
            Parameters
            ----------
            self : object
                Perceptron instance
            Returns
            -------
            new inputs : 2d list
                input including bias node
        '''
        new_inputs = [row[:] for row in self.X]
        [row.append(-1) for row in new_inputs]
        return new_inputs

    def feed_forward(self):
        ''' Computes new activations, acts as thresholding function
            Parameters
            ----------
            self : object
                Perceptron instance
            Returns
            -------
            activations : 2d list
                Activations after thresholding
        '''
        activations = [[0]*self.target_dims for _ in range(self.num_vectors)]
        input_plus_bias = self.input_with_bias()

        for i in range(self.num_vectors):
            for j in range(self.target_dims):
                for k in range(self.input_dims + 1):
                    activations[i][j] += self.weights[k][j]*input_plus_bias[i][k]
                
                if activations[i][j] > 0:
                    activations[i][j] = 1
                else:
                    activations[i][j] = 0

        return activations

    def dot(self, a, b):
        ''' Calculates product of two 2d lists
            Parameters
            ----------
            self : object
                Perceptron instance
            a : 2d list
                Multiplicand of shape (x, y)
            b : 2d list
                Multiplier of shape (y, z)
            Returns
            -------
            ret : 2d list
                Product of ab as shape (x, z)
        '''
        ret = [[0]*len(b[0]) for _ in range(len(a))]

        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(b)):
                    ret[i][j] += a[i][k]*b[k][j]

        return ret

    def transpose(self, arr):
        ''' Transpose a 2d list, i.e. take from shape (a, b) to (b, a)
            ----------
            self : object
                Perceptron instance
            arr : 2d list
                2d list to be transposed
            Returns
            -------
            res: 2d list
                Transposed 2d list
        '''
        res = [[0]*len(arr) for _ in range(len(arr[0]))]

        for i in range(len(arr)):
            for j in range(len(arr[0])):
                res[j][i] = arr[i][j]

        return res

    def subtract(self, a, b):
        ''' Subtract an 2d list from another 2d list element-wise
            Parameters
            ----------
            self : object
                Perceptron instance
            a : 2d list
                minuend
            b : string
                subtrahend
            Returns
            -------
            res : 2d list
                Result of a - b
        '''
        res = [[0]*len(a[0]) for _ in range(len(a))]

        for i in range(len(res)):
            for j in range(len(res[0])):
                res[i][j] = a[i][j] - b[i][j]

        return res

    def multiply(self, a, b):
        ''' Multiply a 2d list by a scalar value element-wise
            Parameters
            ----------
            self : object
                Perceptron instance
            a : 2d list
                multiplicand
            b : number
                multiplier
            Returns
            -------
            res : 2d list
                Result of ab
        '''
        res = [[0]*len(a[0]) for _ in range(len(a))]

        for i in range(len(res)):
            for j in range(len(res[0])):
                res[i][j] = a[i][j]*b
        
        return res

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
        for i in range(iterations - 1):
            self.weights = self.subtract(self.weights, self.multiply(self.dot(self.transpose(self.input_with_bias()), self.subtract(self.activations, self.y)), eta))
            self.activations = self.feed_forward()

    def confusion_matrix(self, inputs, targets):
        ''' Prints confusion matrix to terminal
            Parameters
            ----------
            self : object
                Perceptron instance
            Returns
            -------
            None
        '''
        inputs = np.concatenate((inputs, -np.ones((self.num_samples, 1))), axis=1)
        outputs = np.dot(inputs, self.weights)
        num_classes = np.shape(targets)[1]

        if num_classes == 1:
            num_classes = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                matrix[i, j] = np.sum(np.where(outputs == i, 1, 0)*np.where(targets == j, 1, 0))

        print(matrix)
        print(np.trace(matrix)/np.sum(matrix))
