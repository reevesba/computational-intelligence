''' Magic Square Individual

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from magic_square.ms_fitness import MagicSquareFitness
from sklearn.utils import shuffle
from typing import List, TypeVar

# Custom types
Square = TypeVar("Square")

class Square:
    def __init__(self: Square, length: int, values: List, parent_a: Square, parent_b: Square) -> None:
        ''' Initialize Square instance
            Parameters
            ----------
            self : Square instance
            length : Square length
            values : Square values
            parent_a : Square's first parent
            parent_b : Square's second parent

            Returns
            -------
            None
        '''
        self.length = length
        if values: self.values = values
        else: self.values = self.__random_square()
        self.parent_a = parent_a
        self.parent_b = parent_b
        self.fitness_function = MagicSquareFitness(self.length)
        self.fitness = self.fitness_function.fitness(self.values)
        
    def __random_square(self: Square) -> List:
        ''' Generate random square values
            Parameters
            ----------
            self : Square instance

            Returns
            -------
            Random square values
        '''
        return shuffle([i + 1 for i in range(self.length)])

    def get_values(self: Square) -> List:
        ''' Return square values
            Parameters
            ----------
            self : Square instance

            Returns
            -------
            Square values
        '''
        return self.values

    def get_parent_a(self: Square) -> Square:
        ''' Return square's first parent
            Parameters
            ----------
            self : Square instance

            Returns
            -------
            Square's first parent
        '''
        return self.parent1
    
    def get_parent_b(self: Square) -> Square:
        ''' Return square's second parent
            Parameters
            ----------
            self : Square instance

            Returns
            -------
            Square's second parent
        '''
        return self.parent2

    def get_fitness(self: Square) -> int:
        ''' Return square's fitness score
            Parameters
            ----------
            self : Square instance

            Returns
            -------
            Square's fitness
        '''
        return self.fitness