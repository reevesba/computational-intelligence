''' 
    About: Simple perceptron implementation using McCulloch
           and Pitts' mathematical model of a neuron. Uses
           numpy package.
    Reference: https://seat.massey.ac.nz/personal/s.r.marsland/MLbook.html
    Author: Bradley Reeves
    Date: 04/08/2021
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
        self.x = inputs
        self.y = targets

        if np.ndim(self.x) > 1:
            self.input_dims = np.shape(self.x)[1]
        else:
            self.input_dims = 1

        if np.ndim(self.y) > 1:
            self.target_dims = np.shape(self.y)[1]
        else:
            self.target_dims = 1

        self.num_vectors = np.shape(self.x)[0]
        self.weights = np.random.rand(self.input_dims + 1, self.target_dims)*0.1 - 0.05
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
        return np.concatenate((self.x, -np.ones((self.num_vectors, 1))), axis=1)

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
        for i in range(iterations - 1):
            self.weights -= eta*np.dot(np.transpose(self.input_with_bias()), self.activations - self.y)
            self.activations = self.feed_forward()

    def confusion_matrix(self):
        ''' Prints confusion matrix to terminal
            Parameters
            ----------
            self : object
                Perceptron instance
            Returns
            -------
            None
        '''
        outputs = np.where(np.dot(self.input_with_bias(), self.weights) > 0, 1, 0)
        num_classes = np.shape(self.y)[1]

        if num_classes == 1:
            num_classes = 2
        else:
            outputs = np.argmax(outputs, 1)
            self.y = np.argmax(self.y, 1)

        matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                matrix[i, j] = np.sum(np.where(outputs == i, 1, 0)*np.where(self.y == j, 1, 0))

        print(matrix)
        print(np.trace(matrix)/np.sum(matrix))

def main():
    # Set hyperparameters
    eta = 0.25
    iterations = 10

    # Create test sets
    test_set_a = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    test_set_b = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    # Execute tests
    model_a = Perceptron(test_set_a[:, 0:2], test_set_a[:, 2:])
    model_a.train(eta, iterations)
    model_a.confusion_matrix()

    model_b = Perceptron(test_set_b[:, 0:2], test_set_b[:, 2:])
    model_b.train(eta, iterations)
    model_b.confusion_matrix()

if __name__ == "__main__": 
    main()
