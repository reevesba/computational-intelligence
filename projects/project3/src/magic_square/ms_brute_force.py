''' Magic Square Brute Force Solution
    for 3x3 square

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from numpy import random, sqrt, power
from sklearn.utils import shuffle
from functools import reduce
from typing import Union, List, TypeVar

# Square types
MagicSquareBF = TypeVar("MagicSquareBF")

class MagicSquareBF:
    def __init__(self: MagicSquareBF, length: int) -> None:
        ''' Initialize MagicSquareBF instance
            Parameters
            ----------
            self : MagicSquareBF instance
            length : Unrolled magic square length

            Returns
            -------
            None
        '''
        self.square = shuffle([i + 1 for i in range(length)])
        self.length = len(self.square)
        self.sqrt_length = int(sqrt(self.length))
        self.magic_nbr = (self.sqrt_length + power(self.sqrt_length, 3))/2

    def __swap_values(self: MagicSquareBF) -> None:
        ''' Swap two values in the square list
            Parameters
            ----------
            self : MagicSquareBF instance

            Returns
            -------
            None
        '''
        # Get the swap indexes
        indexes = set()
        while len(indexes) < 2:
            indexes.add(random.randint(self.length))
        indexes = list(indexes)

        # Do swap
        self.square[indexes[0]], self.square[indexes[1]] = self.square[indexes[1]], self.square[indexes[0]]

    def __get_sum(self: MagicSquareBF, square: List, indexes: List) -> int:
        ''' Sum elements of square given indexes
            Parameters
            ----------
            self : MagicSquareBF instance
            square : List of square values
            indexes : List of square indexes

            Returns
            -------
            Calculated sum
        '''
        return reduce(lambda x, y: x + y, [square[i] for i in indexes])

    def __get_diag_indexes(self: MagicSquareBF) -> List:
        ''' Get the indexes of the square diagonals
            Parameters
            ----------
            self : MagicSquareBF instance

            Returns
            -------
            indexes_a, indexes_b : diagonal indexes
        '''
        step_a = self.sqrt_length + 1
        step_b = self.sqrt_length - 1

        indexes_a = [i for i in range(0, self.length, step_a)]
        indexes_b = [i for i in range(step_b, self.length - step_b, step_b)]
        return [indexes_a, indexes_b]

    def __validate_square(self: MagicSquareBF) -> bool:
        ''' Check if magic square has been created
            Parameters
            ----------
            self : MagicSquareBF instance

            Returns
            -------
            True if valid magic square, False otherwise
        '''
        # Check rows
        for i in range(0, self.length, self.sqrt_length):
            indexes = [i + j for j in range(self.sqrt_length)]
            if self.magic_nbr != self.__get_sum(self.square, indexes):
                return False

        # Check columns
        for i in range(self.sqrt_length):
            indexes = [i + j for j in range(0, self.length, self.sqrt_length)]
            if self.magic_nbr != self.__get_sum(self.square, indexes):
                return False

        # Check diagonals
        for indexes in self.__get_diag_indexes():
            if self.magic_nbr != self.__get_sum(self.square, indexes):
                return False

        return True

    def get_magic_square(self: MagicSquareBF) -> List:
        ''' Create a magic square
            Parameters
            ----------
            self : MagicSquareBF instance

            Returns
            -------
            A magic square
        '''
        while True:
            self.__swap_values()
            if self.__validate_square(): 
                return self.square
