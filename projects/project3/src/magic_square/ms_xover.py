''' Magic Square Crossover Function

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from numpy import random
from typing import List, TypeVar

# Custom types
Square = TypeVar("Square")
MagicSquareCrossover = TypeVar("MagicSquareCrossover")

class MagicSquareCrossover:
    def __init__(self: MagicSquareCrossover, length: int) -> None:
        ''' Initialize MagicSquareCrossover instance
            Parameters
            ----------
            self : MagicSquareCrossover instance
            length : Length of square

            Returns
            -------
            None
        '''
        self.length = length

    def __get_inv_seq(self: MagicSquareCrossover, parent: Square) -> List:
        ''' Convert square to inversion sequence
            Parameters
            ----------
            self : MagicSquareCrossover instance
            parent : One parent selected for breeding

            Returns
            -------
            inv_seq : The inversion sequence of the square
        '''
        square = parent.get_square()
        inv_seq = [0 for _ in square]

        for i in range(len(square)):
            for j in range(square.index(i + 1)):
                if square[j] > i + 1: inv_seq[i] += 1
        return inv_seq

    def __do_xover(self: MagicSquareCrossover, inv_a: List, inv_b: List, xover_point: int) -> List:
        ''' Cross two inversion sequences
            Parameters
            ----------
            self : MagicSquareCrossover instance
            inv_a : First inversion sequence
            inv_b : Second inversion sequence
            xover_point : The crossover point

            Returns
            -------
            new_sequence : New inversion sequence
        '''
        new_sequence = []
        for i in range(self.length):
            if i <= xover_point: new_sequence.append(inv_a[i])
            else: new_sequence.append(inv_b[i])
        return new_sequence

    def __rev_inv_seq(self: MagicSquareCrossover, inv_seq: List) -> List:
        ''' Reverse an inversion sequence
            Parameters
            ----------
            self : MagicSquareCrossover instance
            inv_seq : Inversion sequence to reverse

            Returns
            -------
            square : Reversed inversion sequence
        '''
        position = [0 for _ in inv_seq]
        square = [0 for _ in inv_seq]

        for i in range(len(inv_seq) - 1, -1, -1):
            position[i] = inv_seq[i]
            for j in range(i + 1, len(inv_seq)):
                if position[j] >= inv_seq[i]: position[j] += 1

        for i in range(len(inv_seq)):
            square[position[i]] = i + 1
        return square

    def xover(self: MagicSquareCrossover, parents: List) -> List:
        ''' Single point crossover
            ----------
            self : MagicSquareCrossover instance
            parents : Parents to crossover

            Returns
            -------
            New children
        '''
        # Build the inversion sequences
        inv_a = self.__get_inv_seq(parents[0])
        inv_b = self.__get_inv_seq(parents[1])

        # Get the crossover point
        xover_point = random.randint(0, self.length)

        # Do crossover
        inv_a_crossed = self.__do_xover(inv_a, inv_b, xover_point)
        inv_b_crossed = self.__do_xover(inv_b, inv_a, xover_point)

        # Return children
        return [self.__rev_inv_seq(inv_a_crossed), self.__rev_inv_seq(inv_b_crossed)]