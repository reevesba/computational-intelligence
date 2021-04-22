''' Logic function example

Author: Bradley Reeves
Date:   04/21/2021

Code adapted from Chapter 4 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np
from mlp import MLP

def main():
    # Two dimensions
    and_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    model_a = MLP(and_data[:, 0:2], and_data[:, 2:3], num_nodes=2)
    model_a.train(and_data[:, 0:2], and_data[:, 2:3], eta=0.25, epochs=1001)
    model_a.confmat(and_data[:, 0:2], and_data[:, 2:3])

    model_b = MLP(xor_data[:, 0:2], xor_data[:, 2:3], num_nodes=2, out_type='logistic')
    model_b.train(xor_data[:, 0:2], xor_data[:, 2:3], eta=0.25, epochs=5001)
    model_b.confmat(xor_data[:, 0:2], xor_data[:, 2:3])

    # Three dimensions
    and_data = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]])
    xor_data = np.array([[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]])

    model_a = MLP(and_data[:, 0:2], and_data[:, 2:4], num_nodes=2, out_type='linear')
    model_a.train(and_data[:, 0:2], and_data[:, 2:4], eta=0.25, epochs=1001)
    model_a.confmat(and_data[:, 0:2], and_data[:, 2:4])

    model_b = MLP(xor_data[:, 0:2], xor_data[:, 2:4], num_nodes=2, out_type='linear')
    model_b.train(xor_data[:, 0:2], xor_data[:, 2:4], eta=0.15, epochs=5001)
    model_b.confmat(xor_data[:, 0:2], xor_data[:, 2:4])

if __name__ == "__main__":
    main()
