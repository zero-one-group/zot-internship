"""How many such routes are there through a 20×20 grid?"""

import math


def count_path(grid):
    return int(math.factorial(grid*2)) / (math.factorial(grid)**2)


count_path(20)
