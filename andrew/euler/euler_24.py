'''
What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?
'''

import itertools

perm = list(itertools.permutations(range(10), 10))
print(perm[999999])
