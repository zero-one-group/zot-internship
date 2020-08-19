"""The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
Find the sum of all the primes below two million."""

import math


def is_prime(number):
    if number == 2:
        return True
    if number == 1 or number % 2 == 0:
        return False 
    for i in range(3, (int(math.sqrt(number)) + 1), 2):
        if number % i == 0:
            return False;
    return True  


sum(i for i in range(2000000) if is_prime(i))
