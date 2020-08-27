'''
Starting in the top left corner of a 2×2 grid, and only being able to move to the right and down, there are exactly 6 routes to the bottom right corner.How many such routes are there through a 20×20 grid?
Note: This can be reduced down to binary problem. The number of routes can be calculated by (2n)C(n) where n is the grid size.
'''

import math

def path_counter(grid_size):
    return math.factorial(grid_size*2) / (math.factorial(grid_size)**2)
    

n = path_counter(20)
print(int(n), "routes found")
