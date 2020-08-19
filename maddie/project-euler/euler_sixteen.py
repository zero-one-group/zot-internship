#!/usr/bin/env python

number = list(str(2**1000))
number = [int(element) for element in number]

power_digit_sum = sum(number)

print(power_digit_sum)

