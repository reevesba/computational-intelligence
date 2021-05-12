''' Magic Square Fitness Calculator

Author: Bradley Reeves, Sam Shissler
Date:   05/08/2021

Better fitness scores are closer to 0. 
'''

from numpy import power, abs, sqrt
from functools import reduce
from typing import List, TypeVar

MagicSquareFitness = TypeVar("MagicSquareFitness")

class MagicSquareFitness:
    def __init__(self: MagicSquareFitness, length: int) -> None:
        ''' Initialize MagicSquareFitness Instance
            Parameters
            ----------
            self : MagicSquareFitness instance
            length : Length of flattened square

            Returns
            -------
            None
        '''
        self.length = length
        self.sqrt_length = int(sqrt(self.length))
        self.magic_nbr = (self.sqrt_length + power(self.sqrt_length, 3))/2

    def __get_sum(self: MagicSquareFitness, square: List, indexes: List) -> int:
        ''' Sum elements of square given indexes
            Parameters
            ----------
            self : MagicSquareFitness instance
            square : List of square values
            indexes : List of square indexes

            Returns
            -------
            Calculated sum
        '''
        return reduce(lambda x, y: x + y, [square[i] for i in indexes])

    def __get_diag_indexes(self: MagicSquareFitness) -> List:
        ''' Get the indexes of the square diagonals
            Parameters
            ----------
            self : MagicSquareFitness instance

            Returns
            -------
            indexes_a, indexes_b : diagonal indexes
        '''
        step_a = self.sqrt_length + 1
        step_b = self.sqrt_length - 1

        indexes_a = [i for i in range(0, self.length, step_a)]
        indexes_b = [i for i in range(step_b, self.length - step_b, step_b)]
        return [indexes_a, indexes_b]

    def __get_diags_fitness(self: MagicSquareFitness, square: List) -> int:
        ''' Calculate the fitness of the diagonals
            Parameters
            ----------
            self : MagicSquareFitness instance
            square : List of square values

            Returns
            -------
            fitness : Fitness of diagonals
        '''
        fitness = 0
        for indexes in self.__get_diag_indexes():
            fitness += abs(self.magic_nbr - self.__get_sum(square, indexes))
        return fitness

    def __get_rows_fitness(self: MagicSquareFitness, square: List) -> int:
        ''' Calculate the fitness of the rows
            Parameters
            ----------
            self : MagicSquareFitness instance
            square : List of square values

            Returns
            -------
            fitness : Fitness of rows
        '''
        fitness = 0
        for i in range(0, self.length, self.sqrt_length):
            indexes = [i + j for j in range(self.sqrt_length)]
            fitness += abs(self.magic_nbr - self.__get_sum(square, indexes))
        return fitness

    def __get_cols_fitness(self: MagicSquareFitness, square: List) -> int:
        ''' Calculate the fitness of the columns
            Parameters
            ----------
            self : MagicSquareFitness instance
            square : List of square values

            Returns
            -------
            fitness : Fitness of columns
        '''
        fitness = 0
        for i in range(self.sqrt_length):
            indexes = [i + j for j in range(0, len(square), self.sqrt_length)]
            fitness += abs(self.magic_nbr - self.__get_sum(square, indexes))
        return fitness

    def fitness(self: MagicSquareFitness, square: List) -> int:
        ''' Calculate the fitness of the square
            Parameters
            ----------
            self : MagicSquareFitness instance
            square : List of square values

            Returns
            -------
            Fitness of square
        '''
        return (self.__get_diags_fitness(square) 
              + self.__get_rows_fitness(square)
              + self.__get_cols_fitness(square))