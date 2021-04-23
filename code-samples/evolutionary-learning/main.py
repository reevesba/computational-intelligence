''' Evolutionary Learning Demo Driver

Author: Bradley Reeves
Date : 04/22/2021
'''

import utilities
from knapsack_exhaust import exhaustive_search
from knapsack_greedy import greedy_search

def main():
    item_sizes = utilities.get_item_sizes()

    ''' Solving the knapsack problem w/ different approaches
    '''

    # Test brute-force method
    #exhaustive_search(item_sizes)

    # Test greedy method
    #greedy_search(item_sizes)

    # Test genetic algorithm

if __name__ == "__main__":
    main()