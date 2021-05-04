''' Sinewave regression example

Author: Bradley Reeves, Sam Shissler
Date:   04/21/2021

'''

import matplotlib.pyplot as plt
import numpy as np
from os import path
from mlp import MLP
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

def f(x, y):
    ''' Compute sine function of two variables
        ----------
        x : 2d numpy array
            Collection of input vectors x₁, x₂, ..., xₙ
        y : 2d numpy array
            Collection of input vectors y₁, y₂, ..., yₙ
        Returns
        -------
        2d numpy array
            Collection of output vectors z₁, z₂, ..., zₙ
    '''
    return np.sin(np.pi*10*x + 10/(1 + y**2)) + np.log(x**2 + y**2)

def gen_dataset(train_path, test_path):
    ''' Generate dataset for regression testing
        ----------
        train_path : string
            Path to store training data
        test_path : string
            Path to store test data
        Returns
        -------
        None
    '''
    x = np.linspace(1, 100, 50).reshape((50, 1))
    y = np.linspace(1, 100, 50).reshape((50, 1))
    z = f(x, y)
    dataset = np.dstack((x, y, z)).reshape((50, 3))

    # Train/test split
    split = int(np.round(len(dataset)*0.6))
    np.random.shuffle(dataset)

    np.savetxt(train_path, dataset[:split, :], delimiter=',', header='x,y,z', comments='')
    np.savetxt(test_path, dataset[split + 1:, :], delimiter=',', header='x,y,z', comments='')

def plot_best_fit(dataset, X_test, prediction, implementation, n_nodes, n_layers):
    ''' 3d plot showing best fit of all model architectures
        ----------
        dataset : 2d numpy array
            Entire dataset, sorted
        X_test : 2d numpy array
            Collection of input vectors x₁, x₂, ..., xₙ
        prediction : 2d numpy array
            Output results of model
        implementation : string
            Model implementation
        n_nodes : integer
            Number of nodes in hidden layer(s)
        n_layers: integer
            Number of hidden layers
        Returns
        -------
        None
    '''
    # Split the test input nodes
    X_plot = X_test[:, :1].reshape((len(X_test), 1))
    Y_plot = X_test[:, 1:2].reshape((len(X_test), 1))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Surface plot showing original function
    x_surface, y_surface = np.meshgrid(dataset[:, :1], dataset[:, 1:2])
    ax.plot_surface(x_surface, y_surface, dataset[:, 2:3], cmap='gist_heat', alpha=0.75)

    # Superimposed scatter plot showing predictions
    ax.scatter(X_plot, Y_plot, prediction, c='#000')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Function Estimation w/ MLP')
    ax.view_init(12, 5)
    plt.plot([], [], ' ', label='Surface plot = function')
    plt.plot([], [], ' ', label='Scatter plot = estimate')
    plt.legend(loc='upper center')
    plt.savefig('../out/sinewave_test_' + implementation + str(n_nodes) + str(n_layers) + '.png')
    plt.close(fig)

def plot_errors(x, y, n_layers, implementation):
    ''' Confustion matrix for classification
        ----------
        x : list
            Number of hidden nodes tested
        y : list
            Errors calculated for given nodes
        n_layers : integer
            Number of hidden layers used in model
        implementation : string
            Model implementation
        Returns
        -------
        None
    '''
    fig = plt.figure()
    ax = plt.axes()
    ax.ticklabel_format(useOffset=False)

    ax.plot(x, y)
    ax.set_title('Training w/ ' + str(n_layers) + ' Layer(s) (' + implementation + ')')
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('RMSE')
    plt.savefig('../out/errors_' + implementation + '_' + str(n_layers))
    plt.close(fig)

def experiment(X_train, y_train, X_test, y_test, implementation, plotting_set):
    ''' Confustion matrix for classification
        ----------
        X_train : 2d numpy array
            Collection of input vectors x₁, x₂, ..., xₙ
        y_train : 2d numpy array
            Supervised learning labels
        X_test : 2d numpy array
            Collection of input vectors x₁, x₂, ..., xₙ
        y_test : 2d numpy array
            Supervised learning labels
        implementation : string
            Model implementation
        plotting_set : 2d numpy array
            Entire dataset, sorted
        Returns
        -------
        None
    '''
    # Initialize hyper-parameters
    eta = 0.025
    epochs = 500
    n_layers_list = [1, 2]
    n_nodes_list = [1, 2, 3, 4, 5, 10, 15, 20]

    # Store results
    minimum_error = np.inf
    best_predictions = []
    best_n_nodes = 0
    best_n_layers = 0

    ''' Test custom implementation
    '''
    if implementation == 'custom':
        for n_layers in n_layers_list:
            all_errors = []
            for n_nodes in n_nodes_list:
                model = MLP(X_train, y_train, hidden_layer_sizes=(n_nodes,)*int(n_layers), activation='linear')
                model.train(X_train, y_train, eta=eta, epochs=epochs)
                X_test_bias = np.concatenate((X_test, -np.ones((np.shape(X_test)[0], 1))), axis=1)
                predictions = model.feed_forward(X_test_bias)

                error = np.sqrt(mean_squared_error(y_test, predictions))
                all_errors.append(error)

                if error < minimum_error:
                    minimum_error = error
                    best_predictions = predictions
                    best_n_nodes = n_nodes
                    best_n_layers = n_layers
                
            plot_errors(n_nodes_list, all_errors, n_layers, implementation)
        plot_best_fit(plotting_set, X_test, best_predictions, implementation, best_n_nodes, best_n_layers)
        print("Best Architecture (custom): num nodes =", best_n_nodes, ", num layers =", best_n_layers, ", RMSE =", minimum_error)
        return

    ''' Test sklearn implementation
    '''
    if implementation == 'sklearn':
        for n_layers in n_layers_list:
            all_errors = []
            for n_nodes in n_nodes_list:
                model = MLPRegressor(hidden_layer_sizes=(n_nodes,)*int(n_layers), activation='logistic', solver='lbfgs', max_iter=epochs)
                model.fit(X_train, y_train.ravel())
                predictions = model.predict(X_test)

                error = np.sqrt(mean_squared_error(y_test, predictions))
                all_errors.append(error)

                if error < minimum_error:
                    minimum_error = error
                    best_predictions = predictions
                    best_n_nodes = n_nodes
                    best_n_layers = n_layers

            plot_errors(n_nodes_list, all_errors, n_layers, implementation)
        plot_best_fit(plotting_set, X_test, best_predictions, implementation, best_n_nodes, best_n_layers)
        print("Best Architecture (sklearn): num nodes =", best_n_nodes, ", num layers =", best_n_layers, ", RMSE =", minimum_error)
        return

    ''' Test keras implementation
    '''
    if implementation == 'keras':
        for n_layers in n_layers_list:
            all_errors = []
            for n_nodes in n_nodes_list:
                model = Sequential()
                model.add(Dense(n_nodes, activation='sigmoid', kernel_initializer='he_uniform'))
                if n_layers == 2:
                    model.add(Dense(n_nodes, activation='sigmoid', kernel_initializer='he_uniform'))
                model.add(Dense(1))
                model.compile(loss='mse', optimizer='adam')
                model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=0)
                predictions = model.predict_step(X_test)

                error = np.sqrt(mean_squared_error(y_test, predictions))
                all_errors.append(error)

                if error < minimum_error:
                    minimum_error = error
                    best_predictions = predictions
                    best_n_nodes = n_nodes
                    best_n_layers = n_layers

            plot_errors(n_nodes_list, all_errors, n_layers, implementation)
        plot_best_fit(plotting_set, X_test, best_predictions, implementation, best_n_nodes, best_n_layers)
        print("Best Architecture (keras): num nodes =", best_n_nodes, ", num layers =", best_n_layers, ", RMSE =", minimum_error)

def main():
    train_path = '../dat/sinewave_train.csv'
    test_path = '../dat/sinewave_test.csv'

    if not path.exists(train_path) or not path.exists(test_path):
        gen_dataset(train_path, test_path)

    train_set = np.loadtxt(train_path, delimiter=',', skiprows=1)
    test_set = np.loadtxt(test_path, delimiter=',', skiprows=1)

    X_train, y_train = train_set[:, :2], train_set[:, 2:]
    X_test, y_test = test_set[:, :2], test_set[:, 2:]

    # Sort the dataset so that it plots nicely
    plotting_set = np.concatenate((train_set, test_set), axis=0)
    plotting_set = plotting_set[np.argsort(plotting_set[:, 0])]
    print(plotting_set)

    # Experiment w/ different MLP implementations & architectures
    #experiment(X_train, y_train, X_test, y_test, 'custom', plotting_set)
    #experiment(X_train, y_train, X_test, y_test, 'sklearn', plotting_set)
    #experiment(X_train, y_train, X_test, y_test, 'keras', plotting_set)

if __name__ == "__main__":
    main()
