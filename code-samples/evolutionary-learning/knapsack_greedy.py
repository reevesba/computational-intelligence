''' Knapsack Greedy Algorithm Solution

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''
import numpy as np

def greedy_search(item_sizes):
    max_size = 500    

    item_sizes.sort()
    new_sizes = item_sizes[-1:0:-1]
    space = max_size
    
    while len(new_sizes) > 0 and space > new_sizes[-1]:
        # Pick largest item that will fit
        item = np.where(space > new_sizes)[0][0]
        print(new_sizes[item])
        space = space - new_sizes[item]
        new_sizes = np.concatenate((new_sizes[:item], new_sizes[item + 1:]))
        
    print("Size = ", max_size - space)