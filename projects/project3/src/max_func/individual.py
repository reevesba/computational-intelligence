''' Function Maximization Individual

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from numpy import random
from max_func.mf_fitness import MaxFuncFitness

from typing import List, TypeVar

# Custom types
Individual = TypeVar("Individual")

class Individual:
    def __init__(self: Individual, length: int, values: List, parent_a: Individual, parent_b: Individual) -> None:
        ''' Initialize Individual instance
            Parameters
            ----------
            self : Individual instance
            length : Individual length
            values : Individual values
            parent_a : Individual's first parent
            parent_b : Individual's second parent

            Returns
            -------
            None
        '''
        self.length = length
        if values: self.values = values
        else: self.values = self.__random_values()
        self.parent_a = parent_a
        self.parent_b = parent_b
        self.fitness_function = MaxFuncFitness()
        self.fitness, self.result = self.fitness_function.fitness(self.values)

    def __random_values(self: Individual) -> List:
        ''' Generate random Individual values
            Parameters
            ----------
            self : Individual instance

            Returns
            -------
            Random Individual values
        '''
        return [random.randint(3, 11), random.randint(4, 9)]

    def get_values(self: Individual) -> List:
        ''' Return Individual values
            Parameters
            ----------
            self : Individual instance

            Returns
            -------
            Individual values
        '''
        return self.values

    def get_parent_a(self: Individual) -> Individual:
        ''' Return Individual's first parent
            Parameters
            ----------
            self : Individual instance

            Returns
            -------
            Individual's first parent
        '''
        return self.parent_a

    def get_parent_b(self: Individual) -> Individual:
        ''' Return Individual's second parent
            Parameters
            ----------
            self : Individual instance

            Returns
            -------
            Individual's second parent
        '''
        return self.parent_b

    def get_fitness(self: Individual) -> float:
        ''' Return Individual's fitness score
            Parameters
            ----------
            self : Individual instance

            Returns
            -------
            Individual's fitness
        '''
        return self.fitness

    def get_result(self: Individual) -> float:
        ''' Return Individual's function result
            Parameters
            ----------
            self : Individual instance

            Returns
            -------
            Individual's function result
        '''
        return self.result