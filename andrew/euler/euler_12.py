'''
What is the value of the first triangle number to have over five hundred divisors?
Note:All of the factors from 1 to the square root of a number contain exactly one half of all the factors of that number.
'''

import numpy as np
import math

def triangle_numbers(num):
    arr = np.arange(1, num+1)
    tri_num = sum(arr)
    return tri_num

def divisors(num):
    count = 0
    for x in range(1, int(math.sqrt(num))+1):
        if num % x == 0:
            count += 1
    return count*2


count = 1
tri_num = 1
while divisors(tri_num) < 500:
    tri_num = triangle_numbers(count)
    count = count + 1
print(tri_num)
