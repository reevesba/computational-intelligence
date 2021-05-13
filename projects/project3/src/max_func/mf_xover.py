''' Function Maximization Crossover Function

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from typing import List, TypeVar

MaxFuncCrossover = TypeVar("MaxFuncCrossover")

class MaxFuncCrossover:
    def __init__(self: MaxFuncCrossover, length: int) -> None:
        ''' Initialize MaxFuncCrossover instance
            Parameters
            ----------
            self : MaxFuncCrossover instance
            length : Length of square

            Returns
            -------
            None
        '''
        self.length = length

    def xover(self: MaxFuncCrossover, parents: List) -> List:
        ''' Single point crossover
            ----------
            self : MaxFuncCrossover instance
            parents : Parents to crossover

            Returns
            -------
            New children
        '''
        child_a = [parents[0].get_values()[0], parents[1].get_values()[1]]
        child_b = [parents[1].get_values()[0], parents[0].get_values()[1]]
        return [child_a, child_b]