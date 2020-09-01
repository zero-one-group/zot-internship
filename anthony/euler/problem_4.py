"""
A palindromic number reads the same both ways. The largest palindrome made from
the product of two 2-digit numbers is 9009 = 91 × 99.

Find the largest palindrome made from the product of two 3-digit numbers.
"""

# Turn to string, than do negative indexing
def is_palindrome(n):
    return str(n) == str(n)[::-1]

def two_digit_numbers():
    return range(10, 100)

max(
    x*y
    for x in two_digit_numbers()
    for y in two_digit_numbers()
    if is_palindrome(x*y)
)

# max(
    # x*y
    # for x in range(10, 100)
    # for y in range(10, 100)
    # if str(x*y) == str(x*y)[::-1]
# )

def three_digit_numbers():
    return range(100, 1000)

max(
    x*y
    for x in three_digit_numbers()
    for y in three_digit_numbers()
    if is_palindrome(x*y)
)

# Programming is about building abstractions.
