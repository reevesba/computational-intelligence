''' The Genetic Algorithm

Author: Bradley Reeves, Sam Shissler
Date:   05/11/2021

'''

from numpy import mod, reciprocal, random
from sklearn.utils import shuffle
from typing import Union, List, TypeVar

# Square types
Square = TypeVar("Square")
MagicSquareCrossover = TypeVar("MagicSquareCrossover")
MagicSquareMutator = TypeVar("MagicSquareMutator")

# GeneticAlgorithm types
GeneticAlgorithm = TypeVar("GeneticAlgorithm")

# MaxFunction types

# For multiple valid types
# data: Union[str, ET.Element]

class GeneticAlgorithm:
    def __init__(self: GeneticAlgorithm, individual: Square, length: int, xover_func: MagicSquareCrossover, 
                 mutate_func: MagicSquareMutator, generations: int = 100, pop_size: int = 100, 
                 mutation_prob: int = -1, num_elite: int = 2, tournaments: bool = True, max: bool = False) -> None:
        ''' Initialize GeneticAlgorithm instance
            Parameters
            ----------
            self : GeneticAlgorithm instance
            individual : A single sample
            length : Length of individual
            xover_func : Crossover operator
            mutate_func : Mutation operator
            generations : Number of generations  
            pop_size : Size of the population 
            mutation_prob : Mutating probability
            num_elite : Number of samples to keep 
            tournament : Select the best samples from parents and children
            max : Maximize the fitness function

            Returns
            -------
            None
        '''
        if mutation_prob < 0: mutation_prob = 1/length

        self.individual = individual
        self.length = length
        self.xover_func = xover_func(self.length)
        self.mutate_func = mutate_func(self.length, mutation_prob)
        
        # Population size should be even
        if mod(pop_size, 2) == 0:
            self.pop_size = pop_size
        else:
            self.pop_size = pop_size + 1
                    
        self.generations = generations
        self.num_elite = num_elite
        self.tournaments = tournaments

        # Initialize the first generation
        self.population = [self.individual(self.length, None, None, None) for _ in range(self.pop_size)]

    def __elitism(self: GeneticAlgorithm) -> List:
        ''' Select the fittest num_elite from current generation
            ----------
            self : GeneticAlgorithm instance

            Returns
            -------
            num_elite individuals
        '''
        fitness = [(self.population[i].get_fitness(), i) for i in range(self.pop_size)]
        fitness.sort()
        return [self.population[fitness[i][1]] for i in range(self.num_elite)]

    def __fps(self: GeneticAlgorithm) -> List:
        ''' Fitness Proportional Selection
            Parameters
            ----------
            self : GeneticAlgorithm instance

            Returns
            -------
            Parents for breeding
        '''
        fitness_reciprocals = [reciprocal(individual.get_fitness()) for individual in self.population]
        p = fitness_reciprocals/sum(fitness_reciprocals)
        return [random.choice(self.population, p=p), random.choice(self.population, p=p)]

    def __xover(self: GeneticAlgorithm, parents: List) -> List:
        ''' Generate two children by combining parents
            Parameters
            ----------
            self : GeneticAlgorithm Instance
            parents : Parents selected for breeding

            Returns
            -------
            Two children
        '''
        return self.xover_func.xover(parents)

    def __mutate(self: GeneticAlgorithm, children: List) -> List:
        ''' Possibly mutate the children
            Parameters
            ----------
            self : GeneticAlgorithm instance
            children : Individuals to mutate

            Returns
            -------
            The children
        '''
        return self.mutate_func.mutate(children)

    def __tournament(self: GeneticAlgorithm, parents: List, children: List) -> List:
        ''' Tournament mode: select fittest of parents and children
            ----------
            self : GeneticAlgorithm instance
            parents : Parents used for breeding
            child_a : first child generated
            child_b : second child generated

            Returns
            -------
            The two fittest individuals
        '''
        competitors = parents + children

        fitness = [(competitors[i].get_fitness(), i) for i in range(len(competitors))]
        fitness.sort()
        return [competitors[fitness[i][1]] for i in range(2)]

    def __generate_generation(self: GeneticAlgorithm) -> List:
        ''' Create the next generation
            Parameters
            ----------
            self : GeneticAlgorithm instance

            Returns
            -------
            new_population : The generated population
        '''
        new_population = []

        # Transfer the best from the current population
        if self.num_elite > 0: new_population = self.__elitism()

        while len(new_population) < self.pop_size:
            parents = self.__fps()
            children = self.__mutate(self.__xover(parents))

            # Instantiate children
            for i in range(len(children)):
                children[i] = self.individual(self.length, children[i], parents[0], parents[1])

            if self.tournaments:
                winners = self.__tournament(parents, children)
                new_population.append(winners[0])
                new_population.append(winners[1])
            else:
                new_population.append(child_a)
                new_population.append(child_b)
        return new_population

    def execute_ga(self: GeneticAlgorithm) -> None:
        ''' Run the genetic algorithm
            Parameters
            ----------
            self : GeneticAlgorithm instance

            Returns
            -------
            None
        '''
        for i in range(self.generations):
            self.population = self.__generate_generation()
            
            best = self.population[0]
            for ind in self.population:
                if ind.get_fitness() < best.get_fitness():
                    best = ind

            # Stopping condition
            if best.get_fitness() == 0:
                return best.get_square()

            
