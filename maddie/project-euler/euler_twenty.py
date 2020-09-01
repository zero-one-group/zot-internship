#!/usr/bin/env python

from math import factorial

digits_of_factorial = list(str(factorial(100)))
digits_of_factorial = [int(element) for element in digits_of_factorial]

factorial_digit_sum = sum(digits_of_factorial)
print(factorial_digit_sum)
