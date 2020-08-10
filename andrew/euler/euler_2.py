'''
Consider the terms in the Fibonacci sequence whose values do not exceed four million.
Find the sum of the even valued terms
'''

from itertools import takewhile

def fib(prev=1, after=2):
    while True:
        yield prev
        prev, after = after, prev+after


total = sum(num for num in takewhile(lambda num: num < 4000000, fib()) if num % 2 == 0)

print("Sum of the even valued terms of Fibonacci sequence below four million:", total)
