#!/usr/bin/env python
import math

def max_prime_factors(n):
    max_prime = -1
    while n % 2 == 0:
        max_prime = 2
        n >>= 1

    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            max_prime = i
            n = n / i

    if n > 2:
        max_prime = n

    return int(max_prime)

n = 600851475143
print(max_prime_factors(n))

