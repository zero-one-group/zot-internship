'''
The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
Find the sum of all the primes below two million.
'''

import numpy as np

def is_prime(number):
    upper_bound = int(np.ceil(np.sqrt(number + 1)))
    for candidate in range(2, upper_bound + 1):
        if number % candidate == 0:
            return False
    return True

print(sum(x for x in range(1, int(2e6) + 1) if is_prime(x)))

