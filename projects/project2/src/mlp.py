''' Multi Layer Perceptron

Author: Bradley Reeves, Sam Shissler
Date:   04/21/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from scipy.special import expit as logistic_sigmoid

class MLP:
    def __init__(self, X_train, y_train, hidden_layer_sizes=(100,), beta=-1, momentum=0.9, out_type='logistic'):
        ''' Initialize MLP instance
            ----------
            self : object
                MLP instance
            X_train : 2d numpy array
                Collection of input vectors x₁, x₂, ..., xₙ
            y_train : 2d numpy array
                Supervised learning labels
            hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
                The ith element represents the number of neurons in the ith layer
            beta : number
                Activation slope paremeter
            momentum : float
                Helps accelerate gradients in right direction   
            out_type : string
                Output neuron type
            Returns
            -------
            None
        '''
        self.n_samples, self.n_features = X_train.shape
        self.n_outputs = y_train.shape[1]
        self.beta = beta
        self.momentum = momentum
        self.out_type = out_type
        self.layer_units = ([self.n_features] + list(hidden_layer_sizes) + [self.n_outputs])

        # Setup hidden layers/nodes
        self.n_hidden_layers = len(hidden_layer_sizes)
        if self.n_hidden_layers < 1:
            raise Exception("Must have at least one hidden layer.")

        if self.n_hidden_layers > 2:
            raise Exception("Maximum 2 hidden layers allowed.")

        if self.n_hidden_layers == 1:
            # He initialization
            self.weights1 = (np.random.rand(self.layer_units[0] + 1, self.layer_units[1]))*np.sqrt(2/self.layer_units[0])
            self.weights2 = (np.random.rand(self.layer_units[1] + 1, self.layer_units[2]))*np.sqrt(2/self.layer_units[1])
        
        if self.n_hidden_layers == 2:
            # He initialization
            self.weights1 = (np.random.rand(self.layer_units[0] + 1, self.layer_units[1]))*np.sqrt(2/self.layer_units[0])
            self.weights2 = (np.random.rand(self.layer_units[1] + 1, self.layer_units[2]))*np.sqrt(2/self.layer_units[1])
            self.weights3 = (np.random.rand(self.layer_units[2] + 1, self.layer_units[3]))*np.sqrt(2/self.layer_units[2])

    def early_stop(self, X_train, y_train, X_valid, y_valid, eta, epochs=100, trace=False):
        ''' Stops training once local minimum reached by validation set
            ----------
            self : object
                MLP instance
            X_train : 2d numpy array
                Collection of input vectors x₁, x₂, ..., xₙ
            y_train : 2d numpy array
                Supervised learning labels
            X_valid : integer
                Collection of input vectors x₁, x₂, ..., xₙ
            y_valid : integer
                Supervised learning labels
            eta : float
                Learning rate  
            epochs : integer
                Training iterations
            trace : boolean
                Print progress at each step
            Returns
            -------
            new_val_error: 2d numpy array
                Current validation error
        '''
        X_valid = np.concatenate((X_valid, -np.ones((np.shape(X_valid)[0], 1))), axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            if trace: print(count)
            self.train(X_train, y_train, eta, epochs)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            X_valid_out = self.feed_forward(X_valid)
            new_val_error = 0.5*np.sum((y_valid - X_valid_out)**2)
            
        if trace: print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error
    	
    def train(self, X_train, y_train, eta, epochs, trace=False):
        ''' Train the model
            ----------
            self : object
                MLP instance
            X_train : 2d numpy array
                Collection of input vectors x₁, x₂, ..., xₙ
            y_train : 2d numpy array
                Supervised learning labels
            eta : float
                Learning rate  
            epochs : integer
                Training iterations
            trace : boolean
                Print progress at each step
            Returns
            -------
            None
        '''
        X_train = np.concatenate((X_train, -np.ones((self.n_samples, 1))), axis=1)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        if self.n_hidden_layers == 2: 
            updatew3 = np.zeros((np.shape(self.weights3)))
            
        for n in range(epochs):
            self.outputs = self.feed_forward(X_train)

            error = 0.5*np.sum((self.outputs - y_train)**2)
            if (np.mod(n, 100) == 0):
                if trace: print("Iteration: ", n, " Error: ", error)    

            # Different types of output neurons
            if self.out_type == 'linear':
            	deltao = (self.outputs - y_train)/self.n_samples
            elif self.out_type == 'logistic':
            	deltao = self.beta*(self.outputs - y_train)*self.outputs*(1.0 - self.outputs)
            elif self.out_type == 'softmax':
                deltao = (self.outputs - y_train)*(self.outputs*(-self.outputs) + self.outputs)/self.n_samples 
            else:
            	print("error")
            
            # Back propagate
            if self.n_hidden_layers == 1:
                deltah = self.hidden1*self.beta*(1.0 - self.hidden1)*(np.dot(deltao, np.transpose(self.weights2)))

                updatew1 = eta*(np.dot(np.transpose(X_train), deltah[:, :-1])) + self.momentum*updatew1
                updatew2 = eta*(np.dot(np.transpose(self.hidden1), deltao)) + self.momentum*updatew2

                self.weights1 -= updatew1
                self.weights2 -= updatew2
                      
            if self.n_hidden_layers == 2:
                deltah = self.hidden2*self.beta*(1.0 - self.hidden2)*(np.dot(deltao, np.transpose(self.weights3)))

                updatew1 = eta*(np.dot(np.transpose(X_train), deltah[:, :-1])) + self.momentum*updatew1
                updatew2 = eta*(np.dot(np.transpose(self.hidden1), deltao)) + self.momentum*updatew2
                updatew3 = eta*(np.dot(np.transpose(self.hidden2), deltao)) + self.momentum*updatew3

                self.weights1 -= updatew1
                self.weights2 -= updatew2
                self.weights3 -= updatew3
            
    def feed_forward(self, inputs):
        ''' Run the network forward
            ----------
            self : object
                MLP instance
            inputs : 2d numpy array
                Collection of input vectors x₁, x₂, ..., xₙ
            Returns
            -------
            2d numpy array
                Results after activation
        '''
        if self.n_hidden_layers == 1:
            self.hidden1 = np.dot(inputs, self.weights1)
            self.hidden1 = logistic_sigmoid(self.hidden1)
            self.hidden1 = np.concatenate((self.hidden1, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            outputs = np.dot(self.hidden1, self.weights2)

        if self.n_hidden_layers == 2:
            self.hidden1 = np.dot(inputs, self.weights1)
            self.hidden1 = logistic_sigmoid(self.hidden1)
            self.hidden1 = np.concatenate((self.hidden1, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            self.hidden2 = np.dot(self.hidden1, self.weights2)
            self.hidden2 = logistic_sigmoid(self.hidden2)
            self.hidden2 = np.concatenate((self.hidden2, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            outputs = np.dot(self.hidden2, self.weights3)

        # Different types of output neurons
        if self.out_type == 'linear':
        	return outputs
        elif self.out_type == 'logistic':
            return logistic_sigmoid(outputs)
        elif self.out_type == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1)*np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")