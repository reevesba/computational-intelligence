# Genetic Algorithm

## Problem 1

### Objective
Construct a magic square using a genetic algorithm. A magic square of order _n_ is an arrangement of the numbers from 1 to _n²_ in an _n_-by-_n_ matrix, with each number occurring exactly once, so that each row, each column, and each main diagonal has the same sum.

### Requirements
 1. Generate an initial population of magic squares with random values. 
 2. The fitness of each individual square is calculated based on the "flatness", that is, the degree of deviation in the sums of the rows, columns, and diagonals.

### Deliverables
- The code.
- The numerical results.
- The description of the genetic operators used.
- Your conclusions.

## Problem 2

### Objective
Maximize the following function using a genetic algorithm.

    f(𝑥, 𝑦)= sin⁡(π*10*𝑥 + 10/(1 + 𝑦²)) + ln⁡(𝑥² + 𝑦²), 
    where 3 ≤ 𝑥 ≤ 10 and 4 ≤ 𝑦 ≤ 8 are real numbers.

### Requirements
1. Use the following operators: selection, crossover, and mutation.

### Deliverables
- The code.
- The maximum value of _f(x, y)_ and the values of _x_ and _y_ for which this maximum is obtained.
- The evolution of the average _f(x, y)_ value for all generations (as a plot).
- The values of the following parameters: number of generations, population size, mutation probability.
- Your conclusions.

### Directory Structure
- doc: Any relevant documentation/report.
- img: Hosts any images for this README/project.
- out: Program output.
- src: Program source code files.