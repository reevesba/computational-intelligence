''' Population Based Incremental Learning Algorithm

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import pylab as pl
import numpy as np

#import fourpeaks as fF
import knapsack as fF

def PBIL():
	pl.ion()
	
	pop_size = 100
	str_len = 20	
	eta = 0.005
	
	#fitnessFunction = 'fF.fourpeaks'
	fitnessFunction = 'fF.knapsack'
	p = 0.5*np.ones(str_len)
	best = np.zeros(501, dtype=float)

	for count in range(501):
		# Generate samples
		population = np.random.rand(pop_size, str_len)
		for i in range(str_len):
			population[:, i] = np.where(population[:, i] < p[i], 1, 0)

		# Evaluate fitness
		fitness = eval(fitnessFunction)(population)

		# Pick best
		best[count] = np.max(fitness)
		first = np.argmax(fitness)
		fitness[first] = 0
		second = np.argmax(fitness)

		# Update vector
		p  = p*(1 - eta) + eta*((population[first, :] + population[second, :])/2)

		if (np.mod(count, 100) == 0):
			print(count, best[count])

	pl.plot(best,'kx-')
	pl.xlabel('Epochs')
	pl.ylabel('Fitness')
	pl.show()
	#print p

PBIL()
