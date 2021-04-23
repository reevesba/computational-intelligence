''' One Max Fitness Function

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''
import numpy as np

def onemax(pop):
    return np.sum(pop, axis=1)
