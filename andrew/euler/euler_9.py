'''
There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc.
'''


import math
def hypotenuse(a, b):
    return math.sqrt((a**2) + (b**2))

def is_pythagorean_triplet(a, b):
    diagonal = hypotenuse(a, b)
    return diagonal == int(diagonal)

product = [
    a * b * hypotenuse(a, b)
    for a in range(1, 1001)
    for b in range(1, a + 1)
    if is_pythagorean_triplet(a, b) and a + b + hypotenuse(a, b) == 1000
]

print(product)

