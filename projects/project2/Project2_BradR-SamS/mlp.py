''' Multi Layer Perceptron

Author: Bradley Reeves, Sam Shissler
Date:   04/27/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from scipy.special import expit as logistic_sigmoid

class MLP:
    def __init__(self, X_train, y_train, hidden_layer_sizes=(100,), beta=-1, momentum=0.9, activation='logistic'):
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
        self.beta = beta
        self.momentum = momentum
        self.activation = activation
        self.layer_units = ([self.n_features] + list(hidden_layer_sizes) + [y_train.shape[1]])

        # Setup hidden layers/nodes
        if len(self.layer_units) - 2 < 1:
            raise ValueError("Must have at least one hidden layer.")

        if len(self.layer_units) - 2 > 2:
            raise ValueError("Maximum 2 hidden layers allowed.")

        # Initialize weights using He et al. method
        if len(self.layer_units) - 2 == 1:
            self.weights1 = (np.random.rand(self.layer_units[0] + 1, self.layer_units[1]))*np.sqrt(2/self.layer_units[0])
            self.weights2 = (np.random.rand(self.layer_units[1] + 1, self.layer_units[2]))*np.sqrt(2/self.layer_units[1])
        
        if len(self.layer_units) - 2 == 2:
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
            
        for n in range(epochs):
            # Run network forward and calculate output error
            self.outputs = self.feed_forward(X_train)

            if trace:
                error = 0.5*np.sum((self.outputs - y_train)**2)
                if (np.mod(n, 100) == 0):
                    print("Iteration: ", n, " Error: ", error)    

            # Output Errors
            if self.activation == 'linear':
            	deltao = (self.outputs - y_train)/self.n_samples
            elif self.activation == 'logistic':
            	deltao = self.beta*(self.outputs - y_train)*self.outputs*(1.0 - self.outputs)
            elif self.activation == 'softmax':
                deltao = (self.outputs - y_train)*(self.outputs*(-self.outputs) + self.outputs)/self.n_samples 
            else:
            	raise ValueError("No activation function specified.")
            
            self.backpropagate(X_train, deltao, eta)

    def backpropagate(self, X_train, deltao, eta):
        ''' Propagate errors backwards through network
            ----------
            self : object
                MLP instance
            X_train : 2d numpy array
                Collection of input vectors x₁, x₂, ..., xₙ
            deltao : 2d numpy array
                Output layer errors
            eta : float
                Learning rate 
            Returns
            -------
            None
        '''
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        # One hidden layer
        if len(self.layer_units) - 2 == 1:
            deltah1 = self.hidden1*self.beta*(1.0 - self.hidden1)*(np.dot(deltao, np.transpose(self.weights2)))

            updatew1 = eta*(np.dot(np.transpose(X_train), deltah1[:, :-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden1), deltao)) + self.momentum*updatew2

            self.weights1 -= updatew1
            self.weights2 -= updatew2
                    
        # Two hidden layers
        if len(self.layer_units) - 2 == 2:
            updatew3 = np.zeros((np.shape(self.weights3)))

            deltah2 = self.hidden2*self.beta*(1.0 - self.hidden2)*(np.dot(deltao, np.transpose(self.weights3)))
            deltah1 = self.hidden1*self.beta*(1.0 - self.hidden1)*(np.dot(deltah2[:, :-1], np.transpose(self.weights2)))

            updatew1 = eta*(np.dot(np.transpose(X_train), deltah1[:, :-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden1), deltah2[:, :-1])) + self.momentum*updatew2
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
        if len(self.layer_units) - 2 == 1:
            self.hidden1 = np.dot(inputs, self.weights1)
            self.hidden1 = logistic_sigmoid(self.hidden1)
            self.hidden1 = np.concatenate((self.hidden1, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            outputs = np.dot(self.hidden1, self.weights2)

        if len(self.layer_units) - 2 == 2:
            self.hidden1 = np.dot(inputs, self.weights1)
            self.hidden1 = logistic_sigmoid(self.hidden1)
            self.hidden1 = np.concatenate((self.hidden1, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            self.hidden2 = np.dot(self.hidden1, self.weights2)
            self.hidden2 = logistic_sigmoid(self.hidden2)
            self.hidden2 = np.concatenate((self.hidden2, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            outputs = np.dot(self.hidden2, self.weights3)

        # Output activations
        if self.activation == 'linear':
        	return outputs
        elif self.activation == 'logistic':
            return logistic_sigmoid(outputs)
        elif self.activation == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1)*np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            raise ValueError("No activation function specified.")

    def confmat(self, inputs, targets):
        ''' Confustion matrix for classification
            ----------
            self : object
                MLP instance
            inputs : 2d numpy array
                Collection of input vectors x₁, x₂, ..., xₙ
            targets : 2d numpy array
                Supervised learning labels
            Returns
            -------
            None
        '''
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.feed_forward(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)*np.where(targets == j, 1, 0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm)/np.sum(cm)*100)