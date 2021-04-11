''' Single Layer Perceptron Demo on Basic Logic Functions

Author: Bradley Reeves
Date:   04/10/2021

Code adapted from Chapter 3 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from pcn_logic_eg import Perceptron

def main():
    eta = 0.25
    iterations = 6

    # Input
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Targets
    and_y = np.array([[0], [0], [0], [1]])
    or_y = np.array([[0], [1], [1], [1]])
    xor_y = np.array([[0], [1], [1], [0]])

    print("Training AND Logic Function:")
    and_model = Perceptron(X, and_y)
    and_model.train(eta, iterations)

    print("Training OR Logic Function:")
    or_model = Perceptron(X, or_y)
    or_model.train(eta, iterations)

    print("Training XOR Logic Function:")
    xor_model = Perceptron(X, xor_y)
    xor_model.train(eta, iterations)

if __name__ == '__main__':
    main()