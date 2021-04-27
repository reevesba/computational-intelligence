''' Sinewave regression example

Author: Bradley Reeves, Sam Shissler
Date:   04/21/2021

'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os import path
from mlp import MLP
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

def f(x, y):
    return np.sin(np.pi*10*x + 10/(1 + y**2)) + np.log(x**2 + y**2)

def gen_dataset():
    x = np.linspace(1, 100, 50).reshape((50, 1))
    y = np.linspace(1, 100, 50).reshape((50, 1))
    z = f(x, y)
    dataset = np.dstack((x, y, z)).reshape((50, 3))

    # Train/test split
    split = int(np.round(len(dataset)*0.6))
    np.random.shuffle(dataset)

    np.savetxt('../dat/sinewave_train.csv', dataset[:split, :], delimiter=',', header='x,y,z', comments='')
    np.savetxt('../dat/sinewave_test.csv', dataset[split + 1:, :], delimiter=',', header='x,y,z', comments='')

def plot_best_fit(dataset, X_test, prediction, method):
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

    # Configure plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Function Estimation w/ MLP')
    ax.view_init(12, 5)
    plt.plot([], [], ' ', label='Surface plot = function')
    plt.plot([], [], ' ', label='Scatter plot = estimate')
    plt.legend(loc='upper center')
    plt.savefig('../out/sinewave_test_' + method + '.png')
    plt.close(fig)

def plot_errors(x, y, n_layers, implementation):
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
    # Initialize hyper-parameters
    eta = 0.025
    epochs = 500
    n_layers_list = [1, 2]
    n_nodes_list = [1, 2, 3, 4, 5, 10, 15, 20]

    # Store results
    minimum_error = np.inf
    best_predictions = []

    if implementation == 'custom':
        for n_layers in n_layers_list:
            all_errors = []
            for n_nodes in n_nodes_list:
                model = MLP(X_train, y_train, hidden_layer_sizes=(n_nodes,)*int(n_layers), out_type='linear')
                model.train(X_train, y_train, eta=eta, epochs=epochs)
                X_test_bias = np.concatenate((X_test, -np.ones((np.shape(X_test)[0], 1))), axis=1)
                predictions = model.feed_forward(X_test_bias)
                error = np.sqrt(mean_squared_error(y_test, predictions))
                all_errors.append(error)

                if error < minimum_error:
                    minimum_error = error
                    best_predictions = predictions
                
            print(all_errors)
            plot_errors(n_nodes_list, all_errors, n_layers, implementation)
        
        plot_best_fit(plotting_set, X_test, best_predictions, implementation)
        return

    elif implementation == 'sklearn':
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

            plot_errors(n_nodes_list, all_errors, n_layers, implementation)

        plot_best_fit(plotting_set, X_test, best_predictions, implementation)
        return

    elif implementation == 'keras':
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

            plot_errors(n_nodes_list, all_errors, n_layers, implementation)
        
        plot_best_fit(plotting_set, X_test, predictions, implementation)

def main():
    train_path = '../dat/sinewave_train.csv'
    test_path = '../dat/sinewave_test.csv'

    if not path.exists(train_path) or not path.exists(test_path):
        gen_dataset()

    train_set = np.loadtxt(train_path, delimiter=',', skiprows=1)
    test_set = np.loadtxt(test_path, delimiter=',', skiprows=1)

    X_train, y_train = train_set[:, :2], train_set[:, 2:]
    X_test, y_test = test_set[:, :2], test_set[:, 2:]

    # Sort the dataset so that it plots nicely
    plotting_set = np.concatenate((train_set, test_set), axis=0)
    plotting_set = plotting_set[np.argsort(plotting_set[:, 0])]

    # Experiment w/ different MLP implementations
    experiment(X_train, y_train, X_test, y_test, 'custom', plotting_set)
    experiment(X_train, y_train, X_test, y_test, 'sklearn', plotting_set)
    experiment(X_train, y_train, X_test, y_test, 'keras', plotting_set)


if __name__ == "__main__":
    main()
