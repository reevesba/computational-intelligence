''' Multi Layer Perceptron

Author: Bradley Reeves
Date:   04/21/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np

class MLP:
    def __init__(self, X_train, y_train, num_nodes, beta=1, momentum=0.9, out_type='logistic'):
        ''' Initialize MLP instance
            ----------
            self : object
                MLP instance
            X_train : 2d numpy array
                Collection of input vectors x₁, x₂, ..., xₙ
            y_train : 2d numpy array
                Supervised learning labels
            num_nodes : integer
                Number of hidden nodes
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
        # Set up network size
        self.nin = np.shape(X_train)[1]
        self.nout = np.shape(y_train)[1]
        self.ndata = np.shape(X_train)[0]
        self.num_nodes = num_nodes

        self.beta = beta
        self.momentum = momentum
        self.out_type = out_type
    
        # Initialise network
        # weights1: connects input layer to hidden layer
        # weights2: connects hidden layer to ouput layer
        self.weights1 = (np.random.rand(self.nin + 1, self.num_nodes) - 0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.num_nodes + 1, self.nout) -0.5)*2/np.sqrt(self.num_nodes)

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
        # Add the inputs that match the bias node
        X_train = np.concatenate((X_train, -np.ones((self.ndata, 1))), axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        for n in range(epochs):
            self.outputs = self.feed_forward(X_train)

            error = 0.5*np.sum((self.outputs - y_train)**2)
            if (np.mod(n, 100) == 0):
                if trace: print("Iteration: ", n, " Error: ", error)    

            # Different types of output neurons
            if self.out_type == 'linear':
            	deltao = (self.outputs - y_train)/self.ndata
            elif self.out_type == 'logistic':
            	deltao = self.beta*(self.outputs - y_train)*self.outputs*(1.0 - self.outputs)
            elif self.out_type == 'softmax':
                deltao = (self.outputs - y_train)*(self.outputs*(-self.outputs) + self.outputs)/self.ndata 
            else:
            	print("error")
            
            deltah = self.hidden*self.beta*(1.0 - self.hidden)*(np.dot(deltao, np.transpose(self.weights2)))
                      
            updatew1 = eta*(np.dot(np.transpose(X_train), deltah[:, :-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden), deltao)) + self.momentum*updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #X_train = X_train[change,:]
            #y_train = y_train[change,:]
            
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
        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0/(1.0 + np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.weights2)

        # Different types of output neurons
        if self.out_type == 'linear':
        	return outputs
        elif self.out_type == 'logistic':
            return 1.0/(1.0 + np.exp(-self.beta*outputs))
        elif self.out_type == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1)*np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self, inputs, targets):
        ''' Prediction (confusion matrix)
            ----------
            self : object
                MLP instance
            inputs : 2d numpy array
                Collection of input vectors x₁, x₂,...,xₙ
            targets : 2d numpy array
                Supervised learning labels
            Returns
            -------
            None
        '''
        # Add the inputs that match the bias node
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

        print("Confusion matrix is: ")
        print(cm)
        print("Percentage Correct: ", np.trace(cm)/np.sum(cm)*100)
