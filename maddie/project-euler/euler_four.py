#!/usr/bin/env python
def is_palindrome(n):
    return str(n) == str(n)[::-1]

def two_digit_numbers():
    return range(10, 100)

max(
    x*y
    for x in two_digit_numbers()
    for y in two_digit_numbers()
    if is_palindrome(x*y)
    
def three_digit_numbers():
    return range(100, 1000)

max(
    x*y
    for x in three_digit_numbers()
    for y in three_digit_numbers()
    if is_palindrome(x*y)
)
