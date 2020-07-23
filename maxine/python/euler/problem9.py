'A Pythagorean triplet is a set of three natural numbers, a < b < c, for which,

a2 + b2 = c2
For example, 32 + 42 = 9 + 16 = 25 = 52.

There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc'

import math

def triplet(n):
    for a in range(n):
        for b in range(1,a):
            c = math.hypot(a,b)
            if c % 1 == 0:
                if a + b + c == n:
                    print (a*b*c)

triplet(1000)

