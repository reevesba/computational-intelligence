''' Magic Square Mutate Function

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from numpy import random
from typing import List, TypeVar

# Custom types
MagicSquareMutator = TypeVar("MagicSquareMutator")

class MagicSquareMutator:
    def __init__(self: MagicSquareMutator, length: int, mutation_prob: float) -> None:
        ''' Initialize MagicSquareMutator instance
            Parameters
            ----------
            self : MagicSquareMutator instance
            length : Square length
            mutation_prob : Probability of mutation occuring

            Returns
            -------
            None
        '''
        self.length = length
        self.mutation_prob = mutation_prob

    def mutate(self: MagicSquareMutator, children: List) -> List:
        ''' Possibly mutate children
            Parameters
            ----------
            self : MagicSquareMutator instance
            children : Mutator candidates

            Returns
            -------
            children : Children with possible mutations
        '''
        for child in children:
            if random.rand() < self.mutation_prob:
                # Generate indexes for swapping
                indexes = set()
                while len(indexes) < 2:
                    indexes.add(random.randint(self.length))
                indexes = list(indexes)

                # Do swap
                child[indexes[0]], child[indexes[1]] = child[indexes[1]], child[indexes[0]]
        return children