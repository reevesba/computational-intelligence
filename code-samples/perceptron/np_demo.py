''' Numpy Functions Demo

Author: Bradley Reeves
Date:   04/10/2021
'''

import numpy as np

def main():
    # np.array: create an array
    test_set_a = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    test_set_b = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    print("np.array: ")
    print(test_set_a)
    print(test_set_b)
    print("")

    # np.ndim: number of array dimensions
    print("np.ndim: ")
    print(np.ndim(test_set_a))
    print(np.ndim(test_set_b))
    print("")

    # np.shape: returns the shape of an array
    print("np.shape: ")
    print(np.shape(test_set_a))
    print(np.shape(test_set_b))
    print("")

    # np.random.rand: random values in a given shape
    print("np.random.rand: ")
    test_set_c = np.random.rand(4, 3)*0.1 - 0.05
    print(test_set_c)
    print("")

    # np.transpose: reverse or permutate the axes of an array; returns the modified array
    #               for an array with two axes, transpose gives the matrix transpose
    print("np.transpose: ")
    test_set_d = np.transpose(test_set_b)
    print(test_set_d)
    print("")

    # np.dot: dot product of two arrays
    print("np.dot: ")
    print(np.dot(test_set_a, test_set_d))
    print("")

    # np.where: return elements chosen from x or y depending on condition
    print("np.where: ")
    print(np.where(test_set_d > 0, 1, 0))
    print("")

    # np.ones: return a new array of given shape and type, filled with ones
    print("np.ones: ")
    test_set_e = -np.ones((len(test_set_a), 1))
    print("")

    # np.concatenate: join a sequence of arrays along an existing axis
    print("np.concatenate: ")
    print(np.concatenate((test_set_a, test_set_e), axis=1))
    print("")

    # np.argmax: returns the indices of the maximum values along an axis
    print("np.argmax: ")
    test_set_f = np.array([[10, 11, 12], [13, 14, 15]])
    print(np.argmax(test_set_f, 1))
    print("")

    # np.trace: return the sum along diagonals of the array
    print("np.trace: ")
    print(test_set_f)
    print(np.trace(test_set_f))
    print("")

    # np.sum: sum of array elements over a given axis
    print("np.sum: ")
    print(np.sum(test_set_f))

if __name__ == "__main__":
    main()