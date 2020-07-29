"""A Pythagorean triplet is a set of three natural numbers, a < b < c, for which,

a2 + b2 = c2
For example, 32 + 42 = 9 + 16 = 25 = 52.

There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc"""
import math

def product_of_pythagorean_triplet(n):
    for a in range(n):
        for b in range(1,a):
            c = math.hypot(a,b)
            if a + b + c == n and c > 0:        #to check for natural numbers
                return int(a*b*c)
product_of_pythagorean_triplet(1000)

