#!/usr/bin/env python

numbers = {1:0}
max = -1
start_max = -1
for i in range(2, 100000):
    n = i
    steps = 0
    while n >= i: 
        if n % 2 == 0:
            n = n/2
        else:
            n = 3*n + 1
        steps = steps + 1

    steps = steps + numbers[n]
    if steps > max:
        max = steps
        start_max = i
    numbers[i] = steps

print(start_max)
