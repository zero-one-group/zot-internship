#!/usr/bin/env python

def isPrime(n):
    return all(n % x != 0 for x in range(2, int(n**.5) + 1))

sum([ 
    y
    for y in range(2, 2000000)
    if isPrime(y)
    ])

