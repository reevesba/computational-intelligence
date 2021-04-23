
''' Knapsack Fitness Function

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''
import numpy as np

def knapsack(population):
    max_size = 500	
    sizes = np.array([109.60, 125.48, 52.16, 195.55, 58.67, 61.87, 92.95, 93.14, 155.05, 110.89, 13.34, 132.49, 194.03, 121.29, 179.33, 139.02, 198.78, 192.57, 81.66, 128.90])

    fitness = np.sum(sizes*population, axis=1)
    fitness = np.where(fitness > max_size, 500 - 2*(fitness - max_size), fitness)
        
    return fitness
