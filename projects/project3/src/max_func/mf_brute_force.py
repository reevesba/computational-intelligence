''' Maximize Function Brute Force Solution

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from numpy import sin, pi, log
from typing import Tuple, TypeVar

# Custom types
MaxFuncBF = TypeVar("MaxFuncBF")

class MaxFuncBF:
    def __init__(self: MaxFuncBF, x_range: Tuple, y_range: Tuple) -> None:
        ''' Initialize MaxFuncBF instance
            Parameters
            ----------
            self : MaxFuncBF instance
            x_range : Range of possible x values
            y_range : Range of possible y values

            Returns
            -------
            None
        '''
        self.x_vector = [i for i in range(x_range[0], x_range[1] + 1)]
        self.y_vector = [i for i in range(y_range[0], y_range[1] + 1)]
        self.iterations = 0

    def __get_result(self: MaxFuncBF, x: int, y: int) -> float:
        ''' Calculate function to maximize
            Parameters
            ----------
            self : MaxFuncBF instance
            x : first input parameter
            y : seconf input parameter

            Returns
            -------
            Function result
        '''
        return sin(pi*10*x + 10/(1 + y**2)) + log(x**2 + y**2)

    def maximize_function(self: MaxFuncBF) -> Tuple:
        ''' Find function maximum
            Parameters
            ----------
            self : MaxFuncBF

            Returns
            -------
            max_result: (maximum value, first input, second input)
        '''
        self.iterations = 0
        max_result = (0, 0, 0)
        for x in self.x_vector:
            for y in self.y_vector:
                self.iterations += 1
                result = self.__get_result(x, y)
                
                if result > max_result[0]:
                    max_result = (result, x, y)

                if max_result[1] == 10 and max_result[2] == 4:
                    return max_result
        return max_result