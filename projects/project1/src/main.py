''' Single Layer Perceptron Demo

Authors: Bradley Reeves, Sam Shissler
Date:    04/16/2021
'''

from sklearn.model_selection import LeaveOneOut
from perceptron import Perceptron
from data_gen import Generator
import numpy as np

def exp1(X, y):
    ''' Test diffent hyperparameters
        Parameters
        ----------
        X : 2d numpy array
            Collection of input vectors x1, x2,...,xn
        y : 2d numpy array
            supervised learning labels
        Returns
        -------
        max_acc : float
            Best accuracy for dataset
    '''
    loocv = LeaveOneOut()
    max_eta = 0
    max_epochs = 0
    max_acc = 0

    # Test diffent learning rates
    for i in range(1, 105, 5):
        eta = i/1000

        # Test diffent epochs
        for j in range(1, 55, 5):
            epochs = j

            # Evaluate using leave one out cross validation
            accuracy = 0
            for train_index, test_index in loocv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = Perceptron(X_train, y_train)
                model.train(eta, epochs)
                accuracy += model.confusion_matrix(X_test, y_test)

            #print("Eta: {:.2}, Epochs: {:} Accuracy: {:.2%}".format(eta, epochs, accuracy/len(dataset)))

            if accuracy/len(X) > max_acc:
                max_eta = eta
                max_epochs = epochs 
                max_acc = accuracy/len(X)

    # Hyperparameters with best performance
    print("Best round: Eta: {:.2}, Epochs: {:} Accuracy: {:.2%}".format(max_eta, max_epochs, max_acc))
    return max_acc

def exp2(X, y, epochs):
    ''' Test diffent hyperparameters
        Parameters
        ----------
        X : 2d numpy array
            Collection of input vectors x1, x2,...,xn
        y : 2d numpy array
            supervised learning labels
        epochs: integer
            number of training iterations
        Returns
        -------
        None
    '''
    etas = [0.01, 0.025, 0.05, 0.075, 0.1]
    for eta in etas:
        # Only concerned with model training
        model = Perceptron(X, y)
        model.train(eta, epochs, True)

def main():
    # Generate 3x3 dataset
    g1 = Generator(3)
    Is = g1.gen_Is(2, 3)
    Ls = g1.gen_Ls(2, 3, 2, 3)
    dataset1 = np.concatenate((Is, Ls), axis=0)
    X1, y1 = dataset1[:, 1:], dataset1[:, :1]

    # Generate 5x5 dataset
    g2 = Generator(5)
    Is = g2.gen_Is(2, 3)
    Ls = g2.gen_Ls(2, 3, 2, 3)
    dataset2 = np.concatenate((Is, Ls), axis=0)
    X2, y2 = dataset2[:, 1:], dataset2[:, :1]

    ''' Experiment 1
        ------------
        Test diffent hyperparameters
        with increasing dimensions
    '''
    max_acc1 = exp1(X1, y1)     # 3x3
    max_acc2 = exp1(X2, y2)     # 5x5


    ''' Experiment 2
        ------------
        Observe how the learning rate
        influences the learning phase
    '''
    exp2(X1, y1, 100)    # 3x3
    exp2(X2, y2, 100)    # 5x5

if __name__ == "__main__":
    main()