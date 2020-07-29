#!/usr/bin/env python

def triangle_number(n):
    return sum ([i for i in range(1, n + 1)])

j = 0
n = 0
count_factors = 0

while count_factors <= 500:
    count_factors = 0
    j += 1
    n = triangle_number(j)

    i = 1
    while i <= n*.5:
        if n % i == 0:
            count_factors += 1
        i += 1

    count_factors *= 2

print(n)
