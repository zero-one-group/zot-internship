"""The prime factors of 13195 are 5, 7, 13 and 29.find largest prime factor of
600851475143"""
import math


def factors_of_n(num):
    return (int(num/x) for x in range(1, num) if num % x == 0)


def prime_factors_of_n(num):
    for factor in factors_of_n(num):
        return (i for i in range(3, int(math.sqrt(num)) + 1) if factor % i == 0 )


max(prime_factors_of_n(600851475143))
