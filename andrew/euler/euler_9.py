'''
There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc.
'''


import math

def hypotenuse(a, b):
    c = math.sqrt((a**2) + (b**2))
    return c


n = 10000
for a in range(1, n+1):
    for b in range(1, n+1):
        c = hypotenuse(a, b)
        if c % 1 == 0:
            tot = a+b+c
            if tot == 1000:
                print("a =", a, "; b =", b, "; c =", c, "; sum=", tot)
                exit()
