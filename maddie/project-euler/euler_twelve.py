#!/usr/bin/env python
#what is the first triangle number to have over 500 divisors

def triangle_number(n):
    return sum([iterator for iterator in range(1, n + 1)])

number = 0
result = 0
count_factors = 0

while count_factors <= 500:
    count_factors = 0
    number += 1
    result = triangle_number(number)

    counter = 1
    while factors <= result*.5:
        if result % counter == 0:
            count_factors += 1
        counter += 1

    count_factors *= 2

print(result)
