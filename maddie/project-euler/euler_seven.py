#!/usr/bin/env python

def is_prime_number(n):
    return all(n % x != 0 for x in range(2, int(n**.5) + 1))

def nth_prime(index):
    number_of_primes = 1
    prime = 2

    while number_of_primes < index:
        prime += 1
        if is_prime_number(prime) == True:
            number_of_primes += 1
    return prime

print(nth_prime(10001))
