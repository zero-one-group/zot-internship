'''
There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc.
'''


import math

# >>> Revised

def hypotenuse(a, b):
    return math.sqrt((a**2) + (b**2))

def is_pythagorean_triplet(a, b):
    diagonal = hypotenuse(a, b)
    return diagonal == int(diagonal)

[
    a * b * hypotenuse(a, b)
    for a in range(1, 1001)
    for b in range(1, a + 1)
    if is_pythagorean_triplet(a, b) and a + b + hypotenuse(a, b) == 1000
]



n = 10000
for a in range(1, n+1):
    for b in range(1, n+1):
        c = hypotenuse(a, b)
        if c % 1 == 0:
            tot = a+b+c
            if tot == 1000:
                print(a * b * c)
                print("a =", a, "; b =", b, "; c =", c, "; sum=", tot)
                # exit()
