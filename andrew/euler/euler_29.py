'''
How many distinct terms are in the sequence generated by ab for 2 ≤ a ≤ 100 and 2 ≤ b ≤ 100?
'''

import numpy as np
from itertools import product

a = list(range(2,101))
b = list(range(2,101))
possibilities = list(product(a, b))

res = [possibilities[idx][0] ** possibilities[idx][1]
        for idx in range(len(possibilities))]

print("Distinct terms =", len(np.unique(res)))
