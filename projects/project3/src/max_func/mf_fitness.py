''' Maximize Function Fitness Calculator

Author: Bradley Reeves, Sam Shissler
Date:   05/08/2021

Better fitness scores are closer to 0. 
'''

from numpy import sin, pi, log, reciprocal
from typing import List, Tuple, TypeVar

MaxFuncFitness = TypeVar("MaxFuncFitness")

class MaxFuncFitness:
    def fitness(self: MaxFuncFitness, values: List) -> Tuple:
        ''' Calculate the fitness of the Individual
            Parameters
            ----------
            self : MaxFuncFitness instance
            values : List of Individual values

            Returns
            -------
            (Fitness of Individual, function result)
        '''
        result = sin(pi*10*values[0] + 10/(1 + values[1]**2)) + log(values[0]**2 + values[1]**2)
        return (reciprocal(result), result)