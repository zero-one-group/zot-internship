"""
2520 is the smallest number that can be divided by each of the numbers
from 1 to 10 without any remainder.

What is the smallest positive number that is evenly divisible
by all of the numbers from 1 to 20?
"""

def is_divisible_by_one_to_twenty(n):
    return all(n % x == 0 for x in range(1, 21))

upper_bound = 2**4 * 3**2 * 5 * 7 * 11 * 13 * 17 * 19

for n in range(1, upper_bound + 1):
    if is_divisible_by_one_to_twenty(n):
        break
n

# Eager: compute everyhing, then take the first one
next([
    n
    for n in range(1, upper_bound + 1)
    if is_divisible_by_one_to_twenty(n)
])

# Lazy: take the first, forget about everything else
next(
    n
    for n in range(1, upper_bound + 1)
    if is_divisible_by_one_to_twenty(n)
)
