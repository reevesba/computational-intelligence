''' Execute the genetic algorithm

Author: Bradley Reeves
Date:   04/22/2021

Code adapted from Chapter 10 of Machine Learning: An Algorithmic 
Perspective (2nd Edition) by Stephen Marsland (http://stephenmonika.net) 
'''

import pylab as pl
from ga import GeneticAlgorithm

pl.ion()
pl.show()

plotfig = pl.figure()

ga = GeneticAlgorithm(str_len=20, fitness_func='fF.knapsack', generations=301)
ga.runGA(plotfig)

#pl.pause(0)
#pl.show()
