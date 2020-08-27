#!/usr/bin/env python
#sum of the squares of the first 100 natural numbers

sum_square = 0
for x in range(1, 101):
    sum_square += x * x
print(sum_square)

#square of the sum of the first 100 natural numbers

square = sum([
    y
    for y in range(1, 101)
    ])
square_sum = square**2

#difference between them
square_sum - sum_square
