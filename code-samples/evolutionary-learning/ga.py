''' The Genetic Algorithm

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import pylab as pl
import numpy as np
#import fourpeaks as fF
import knapsack as fF

class GeneticAlgorithm:
    def __init__(self, str_len, fitness_func, generations, pop_size=100, mutation_prob=-1, crossover='un', num_elite=4, tournament=True):
        ''' Initialize GeneticAlgorithm instance
            ----------
            self : object
                GeneticAlgorithm instance
            str_len : integer
                Length of each sample string
            fitness_func : string
                Python fitness function
            generations : integer
                Evolve for this many generations
            pop_size : integer
                Sample population size
            mutation_prob : signed integer
                Probabilty that string element will mutate   
            crossover : string
                Crossover method to use, can be 'un' for uniform or 'sp' for single point
            num_elite : float
                Number of fittest string to use for mating  
            tournament : boolean
                Use tournement selection
            Returns
            -------
            None
        '''
        self.str_len = str_len
        
        # Population size should be even
        if np.mod(pop_size, 2) == 0:
            self.pop_size = pop_size
        else:
            self.pop_size = pop_size + 1
        
        if mutation_prob < 0:
                self.mutation_prob = 1 / str_len
        else:
                self.mutation_prob = mutation_prob
                    
        self.generations = generations
        self.fitness_func = fitness_func
        self.crossover = crossover
        self.num_elite = num_elite
        self.tournment = tournament

        # Initialize the first generation
        self.population = np.random.rand(self.pop_size, self.str_len)
        self.population = np.where(self.population < 0.5, 0, 1)
        
    def runGA(self, plotfig):
        ''' Initialize GeneticAlgorithm instance
            ----------
            self : object
                GeneticAlgorithm instance
            plotfig : ?
                ?
            Returns
            -------
            None
        '''
        pl.ion()
        #plotfig = pl.figure()
        best_fit = np.zeros(self.generations)

        for i in range(self.generations):
            # Compute fitness of the population
            fitness = eval(self.fitness_func)(self.population)

            # Pick parents -- can do in order since they are randomised
            new_population = self.fps(self.population, fitness)

            # Apply the genetic operators
            # sp: single point
            # un: uniform
            if self.crossover == 'sp':
                new_population = self.spCrossover(new_population)
            elif self.crossover == 'un':
                new_population = self.uniformCrossover(new_population)
            new_population = self.mutate(new_population)

            # Apply elitism and tournaments if using
            if self.num_elite > 0:
                new_population = self.elitism(self.population, new_population, fitness)

            if self.tournament:
                new_population = self.tournament(self.population, new_population, fitness, self.fitness_func)

            self.population = new_population
            best_fit[i] = fitness.max()

            if (np.mod(i, 100) == 0):
                print(i, fitness.max())	

            #pl.plot([i],[fitness.max()],'r+')
        pl.plot(best_fit, 'kx-')
        #pl.show()

    def fps(self, population, fitness):
        ''' Initialize GeneticAlgorithm instance
            ----------
            self : object
                GeneticAlgorithm instance
            population : integer
                Length of each sample string
            fitness : string
                Python fitness function
            Returns
            -------
            new_population : 
        '''
        # Scale fitness by total fitness
        fitness = fitness/np.sum(fitness)
        fitness = 10*fitness/fitness.max()
        
        # Put repeated copies of each string in according to fitness
        # Deal with strings with very low fitness
        j = 0
        while np.round(fitness[j]) < 1:
            j = j + 1
        
        new_population = np.kron(np.ones((int(np.round(fitness[j])), 1)), population[j, :])

        # Add multiple copies of strings into the new_population
        for i in range(j + 1, self.pop_size):
            if np.round(fitness[i]) >= 1:
                new_population = np.concatenate((new_population, np.kron(np.ones((int(np.round(fitness[i])), 1)), population[i, :])), axis=0)

        # Shuffle the order (note that there are still too many)
        indices = np.arange(np.shape(new_population)[0])
        np.random.shuffle(indices)
        new_population = new_population[indices[:self.pop_size], :]
        return new_population	

    def spCrossover(self, population):
        ''' Single point crossover
            ----------
            self : object
                GeneticAlgorithm instance
            population : integer
                Length of each sample string
            Returns
            -------
            new_population : 
        '''
        new_population = np.zeros(np.shape(population))
        crossing_pt = np.random.randint(0, self.str_len, self.pop_size)
        for i in range(0, self.pop_size, 2):
            new_population[i, :crossing_pt[i]] = population[i, :crossing_pt[i]]
            new_population[i + 1, :crossing_pt[i]] = population[i + 1, :crossing_pt[i]]
            new_population[i, crossing_pt[i]:] = population[i + 1, crossing_pt[i]:]
            new_population[i + 1, crossing_pt[i]:] = population[i, crossing_pt[i]:]
        return new_population

    def uniformCrossover(self, population):
        ''' Uniform crossover
            ----------
            self : object
                GeneticAlgorithm instance
            population : integer
                Length of each sample string
            Returns
            -------
            new_population : 
        '''
        new_population = np.zeros(np.shape(population))
        which = np.random.rand(self.pop_size, self.str_len)
        which1 = which >= 0.5
        for i in range(0, self.pop_size, 2):
            new_population[i, :] = population[i, :]*which1[i, :] + population[i + 1, :]*(1 - which1[i, :])
            new_population[i + 1, :] = population[i, :]*(1 - which1[i, :]) + population[i + 1, :]*which1[i, :]
        return new_population
        
    def mutate(self, population):
        ''' Mutation
            ----------
            self : object
                GeneticAlgorithm instance
            population : integer
                Length of each sample string
            Returns
            -------
            population : 
        '''
        where_mutate = np.random.rand(np.shape(population)[0], np.shape(population)[1])
        population[np.where(where_mutate < self.mutation_prob)] = 1 - population[np.where(where_mutate < self.mutation_prob)]
        return population

    def elitism(self, old_population, population, fitness):
        ''' Initialize GeneticAlgorithm instance
            ----------
            self : object
                GeneticAlgorithm instance
            old_population : integer
                Length of each sample string
            population : string
                Python fitness function
            fitness : integer
                Evolve for this many generations
            Returns
            -------
            population : 
        '''
        best = np.argsort(fitness)
        best = np.squeeze(old_population[best[-self.num_elite:], :])
        indices = np.arange(np.shape(population)[0])
        np.random.shuffle(indices)
        population = population[indices, :]
        population[0:self.num_elite, :] = best
        return population

    def tournament(self, old_population, population, fitness, fitness_func):
        ''' Initialize GeneticAlgorithm instance
            ----------
            self : object
                GeneticAlgorithm instance
            old_population : integer
                Length of each sample string
            population : string
                Python fitness function
            fitness : integer
                Evolve for this many generations
            fitness_func : integer
                Sample population size
            Returns
            -------
            population : 
        '''
        new_fitness = eval(self.fitness_func)(population)
        for i in range(0, np.shape(population)[0], 2):
            f = np.concatenate((fitness[i:i + 2], new_fitness[i:i + 2]), axis=0)
            indices = np.argsort(f)
            if indices[-1] < 2 and indices[-2] < 2:
                population[i, :] = old_population[i, :]
                population[i + 1, :] = old_population[i + 1, :]
            elif indices[-1] < 2:
                if indices[0] >= 2:
                    population[i + indices[0] - 2, :] = old_population[i + indices[-1]]
                else:
                    population[i + indices[1] - 2, :] = old_population[i + indices[-1]]
            elif indices[-2] < 2:
                if indices[0] >= 2:
                    population[i + indices[0] - 2, :] = old_population[i + indices[-2]]
                else:
                    population[i + indices[1] - 2, :] = old_population[i + indices[-2]]
        return population
            
