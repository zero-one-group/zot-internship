'''
There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc.
'''


import math

def pythagoras(a, b):
    '''Calculate hypotenuse value'''
    c = math.sqrt((a**2) + (b**2))
    return c


n = 10000
for a in range(1, n+1):
    for b in range(1, n+1):
        c = pythagoras(a, b)
        if c % 1 == 0:
            tot = a+b+c
            if tot == 1000:
                print("a =", a, "; b =", b, "; c =", c, "; sum=", tot)
                exit()
