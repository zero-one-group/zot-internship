"""
If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9.

The sum of these multiples is 23.

Find the sum of all the multiples of 3 or 5 below 1000.
"""

# Two styles of programming: imperative and declarative.

# Imperative style: do this, do that, then do this.
multiples = []
for n in range(1, 1001):
    if n % 3 == 0 or n % 5 == 0:
        multiples.append(n)
sum(multiples)

# Declarative style: just work with definitions
sum([
    x
    for x in range(1, 1001)
    if x % 3 == 0 or x % 5 == 0
])

# List comprehensions vs. generator expressions
sum(
    x
    for x in range(1, 1001)
    if x % 3 == 0 or x % 5 == 0
)

# Vim-Slime
