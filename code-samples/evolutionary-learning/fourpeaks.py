''' Four Peaks Fitness Function

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import numpy as np

def fourpeaks(population):
	T = 15

	start = np.zeros((np.shape(population)[0], 1))
	finish = np.zeros((np.shape(population)[0], 1))
	fitness = np.zeros((np.shape(population)[0], 1))

	for i in range(np.shape(population)[0]):
		s = np.where(population[i, :] == 1)
		f = np.where(population[i, :] == 0)
		if np.size(s) > 0:
			start = s[0][0]
		else:
			start = 0	
		
		if np.size(f) > 0:
			finish = np.shape(population)[1] - f[-1][-1] -1
		else:
			finish = 0

		if start > T and finish > T:
			fitness[i] = np.maximum(start, finish) + 100
		else:
			fitness[i] = np.maximum(start, finish)

	fitness = np.squeeze(fitness)
	return fitness
