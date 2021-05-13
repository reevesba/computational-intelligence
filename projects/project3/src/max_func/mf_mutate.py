''' Function Maximization Mutator

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from numpy import random
from typing import List, TypeVar

MaxFuncMutator = TypeVar("MaxFuncMutator")

class MaxFuncMutator:
    def __init__(self: MaxFuncMutator, length: int, mutation_prob: float) -> None:
        ''' Initialize MaxFuncMutator instance
            Parameters
            ----------
            self : MaxFuncMutator instance
            length : Individual length
            mutation_prob : Probability of mutation occuring

            Returns
            -------
            None 
        '''
        self.length = length
        self.mutation_prob = mutation_prob

    def mutate(self: MaxFuncMutator, children: List) -> List:
        ''' Possibly mutate children
            Parameters
            ----------
            self : MaxFuncMutator instance
            children : Mutator candidates

            Returns
            -------
            children : Children with possible mutations
        '''
        for child in children:
            if random.rand() < self.mutation_prob:
                # Select a random index to mutate
                index = random.randint(self.length)

                # Do mutation
                if index == 0: child[index] = random.randint(3, 11)
                if index == 1: child[index] = random.randint(4, 9)
        return children