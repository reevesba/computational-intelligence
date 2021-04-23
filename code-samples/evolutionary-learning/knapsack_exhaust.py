''' Knapsack Exhaustive Search Solution

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''
import numpy as np

def exhaustive_search(item_sizes):
    best = 0
    twos = np.arange(-len(item_sizes), 0, 1)
    twos = 2.0**twos
    
    for i in range(2**len(item_sizes) - 1):
        string = np.remainder(np.floor(i*twos), 2) 
        fitness = np.sum(string*item_sizes)
        if fitness > best and fitness < 500:
            best = fitness
            best_string = string
    print(best)
    print(best_string)