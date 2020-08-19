#!/usr/bin/env python

#use the central binomial coefficients formula

from math import factorial

def central_binomial(n):
    print(factorial(2*n) / (factorial(n))**2)

central_binomial(20)
