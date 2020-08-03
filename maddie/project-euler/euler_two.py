#!/usr/bin/env python
def fib_numbers():
    last, current = 0, 1
    while current + last < 4e6:
        last, current = current, current + last
        yield current
        
sum(fib for fib in fib_numbers() if fib % 2 == 0)
