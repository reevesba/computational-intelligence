''' Genetic Algorithm Runner

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''
import matplotlib.pyplot as plt
from time import time
from ga import GeneticAlgorithm

# Magic square classes
from magic_square.square import Square
from magic_square.ms_xover import MagicSquareCrossover
from magic_square.ms_mutate import MagicSquareMutator
from magic_square.ms_brute_force import MagicSquareBF

# Maximize function classes
from max_func.individual import Individual
from max_func.mf_xover import MaxFuncCrossover
from max_func.mf_mutate import MaxFuncMutator
from max_func.mf_brute_force import MaxFuncBF

def plot_fitness_evolution(iterations: int, avg_fitness: float, name: str) -> None:
    ''' Plot the average fitness for each generation
        Parameters
        ----------
        avg_fitness : Average fitness for each generation
        iterations : Generations to find optimal values

        Returns
        -------
        None
    '''
    iterations = [i + 1 for i in range(iterations)]

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(iterations, avg_fitness)
    ax.set_title("GA Fitness Evolution")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    plt.savefig("../out/fitevo_" + name + ".png")
    plt.close(fig)

def magic_square_test():
    ''' Magic square experimentation
        Parameters
        ----------
        None

        Returns
        -------
        None
    '''
    # Warning: the brute-force is 
    # VERY inefficient when length > 9
    length = 9

    # Genetic Algorithm
    ga = GeneticAlgorithm(individual=Square, length=9, xover_func=MagicSquareCrossover, mutate_func=MagicSquareMutator, mutation_prob=0.25)
    start_time = time()
    fittest = ga.execute_ga()

    print("Genetic Algorithm:")
    print("Time:", time() - start_time)
    print("Fittest:", fittest.get_values())
    print("Iterations:", ga.iterations)
    plot_fitness_evolution(ga.iterations, ga.avg_fitness, "ms")

    # Brute force algorithm
    bf = MagicSquareBF(length=9)
    start_time = time()
    fittest = bf.get_magic_square()

    print("Brute-Force Algorithm:")
    print("Time:", time() - start_time)
    print("Fittest:", fittest)
    print("Iterations:", bf.iterations)

def max_function_test():
    ''' Maximize function experimentation
        Parameters
        ----------
        None

        Returns
        -------
        None
    '''
    # Genetic Algorithm
    ga = GeneticAlgorithm(individual=Individual, length=2, xover_func=MaxFuncCrossover, mutate_func=MaxFuncMutator, mutation_prob=0.25)
    start_time = time()
    fittest = ga.execute_ga()

    print("Genetic Algorithm:")
    print("Time:", time() - start_time)
    print("Fittest:", fittest.get_result(), fittest.get_values())
    print("Iterations:", ga.iterations)
    plot_fitness_evolution(ga.iterations, ga.avg_fitness, "mf")

    # Brute force algorithm
    bf = MaxFuncBF(x_range=(3, 10), y_range=(4, 8))
    start_time = time()
    fittest = bf.maximize_function()

    print("Brute-Force Algorithm:")
    print("Time:", time() - start_time)
    print("Fittest:", fittest)
    print("Iterations:", bf.iterations)

if __name__ == "__main__":
    print("Magic Squares:")
    magic_square_test()

    print("\nFunction Maximization:")
    max_function_test()
