''' Genetic Algorithm Runner

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from ga import GeneticAlgorithm
from magic_square.square import Square
from magic_square.ms_xover import MagicSquareCrossover
from magic_square.ms_mutate import MagicSquareMutator
from magic_square.ms_brute_force import MagicSquareBF

def main():
    # Task 1: Generate a magic square
    ga_a = GeneticAlgorithm(individual=Square, length=9, xover_func=MagicSquareCrossover, mutate_func=MagicSquareMutator)
    print(ga_a.execute_ga())

    # Test brute-force solution
    bf_a = MagicSquareBF(length=9)
    print(bf_a.get_magic_square())

    # Task 2: Maximize a function
    
if __name__ == "__main__":
    main()
