''' Generates Ls & Is

Authors: Bradley Reeves, Sam Shissler
Date:    04/19/2021

Note: targets setup as 1 (is an L) or 0 (is not an L). This
value is prepended to the sample.

Code adapted from Andrew Dunn's implementation which can be found
at github.com/dunn-cwu/CS557_project1_perceptron/blob/master/input_gen.py.
'''

import numpy as np
import time

MAX_FAILS = 50      # Chances to find unique sample
MAX_TIME = 240      # Max time looking for unique sample

class Generator:
    def __init__(self, size):
        ''' Initialize Generator instance
            ----------
            self : object
                Generator instance
            size : integer
                Square matrix dimensions (size x size)
            Returns
            -------
            None
        '''
        np.random.seed(1)
        self.size = size

    def is_unique(self, all_samples, test_sample):
        ''' Generated sample should be unique
            ----------
            self : object
                Generator instance
            all_samples : 2d numpy array
                Samples generated thus far
            test_sample : numpy array
                Current sample to test
            Returns
            -------
            Boolean
                Whether or not sample is unique
        '''
        for sample in all_samples:
            if np.array_equal(sample, test_sample):
                return False
        return True        

    def gen_I(self, min_height, max_height):
        ''' Generate a single 'I' sample
            ----------
            self : object
                Generator instance
            min_height : integer
                Sample height minimum pixels
            max_height : integer
                Sample height maximum pixels
            Returns
            -------
            sample : numpy array
                Single 'I' sample
        '''
        if max_height > self.size:
            raise Exception("Error: Max height cannot be greater than matrix size.")

        height = np.random.randint(min_height, max_height + 1)
        x_pos = np.random.randint(0, self.size)
        y_pos = np.random.randint(0, self.size - height + 1)

        sample = np.zeros((self.size, self.size))
        for i in range(y_pos, y_pos + height): sample[i][x_pos] = 1

        return np.concatenate((np.array([0]), sample.flatten('C')), axis=0)

    def gen_L(self, min_height, max_height, min_width, max_width):
        ''' Generate a single 'L' sample
            ----------
            self : object
                Generator instance
            min_height : integer
                Sample height minimum pixels
            max_height : integer
                Sample height maximum pixels
            min_width : integer
                Sample width minimum pixels
            max_width : integer
                Sample width maximum pixels
            Returns
            -------
            sample : numpy array
                Single 'L' sample
        '''
        if max_height > self.size:
            raise Exception("Error: Max height cannot be greater than matrix size..")

        if max_width > self.size:
            raise Exception("Error: Max width cannot be greater than matrix size.")
        
        width = np.random.randint(min_width, max_width + 1)
        height = np.random.randint(min_height, max_height + 1)
        x_pos = np.random.randint(0, self.size - width + 1)
        y_pos = np.random.randint(0, self.size - height + 1)

        sample = np.zeros((self.size, self.size))
        for i in range(x_pos + 1, x_pos + width): sample[y_pos + height - 1][i] = 1
        for i in range(y_pos, y_pos + height): sample[i][x_pos] = 1

        return np.concatenate((np.array([1]), sample.flatten('C')), axis=0)

    def gen_Is(self, min_height, max_height):
        ''' Generates as many 'I' samples as possible
            or until stopping condition is met
            ----------
            self : object
                Generator instance
            min_height : integer
                Sample height minimum pixels
            max_height : integer
                Sample height maximum pixels
            Returns
            -------
            samples : 2d numpy array
                All 'I' samples
        '''
        samples = []
        fails = 0
        start_time = time.time()

        while fails < MAX_FAILS:
            sample = self.gen_I(min_height, max_height)

            if self.is_unique(samples, sample):
                fails = 0
                samples.append(sample)
            else:
                fails += 1

            # Stopping condition to prevent infinite loop
            if time.time() - start_time > MAX_TIME: break

        return np.asarray(samples)

    def gen_Ls(self, min_height, max_height, min_width, max_width):
        ''' Generates as many 'L' samples as possible
            or until stopping condition is met
            ----------
            self : object
                Generator instance
            min_height : integer
                Sample height minimum pixels
            max_height : integer
                Sample height maximum pixels
            min_width : integer
                Sample width minimum pixels
            max_width : integer
                Sample width maximum pixels
            Returns
            -------
            sample : 2d numpy array
                All 'L' samples
        '''
        samples = []
        fails = 0
        start_time = time.time()

        while fails < MAX_FAILS:
            sample = self.gen_L(min_height, max_height, min_width, max_width)

            if self.is_unique(samples, sample):
                fails = 0
                samples.append(sample)
            else:
                fails += 1
            
            # Stopping condition to prevent infinite loop
            if time.time() - start_time > MAX_TIME: break

        return np.asarray(samples)