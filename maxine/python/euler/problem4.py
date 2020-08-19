"""Find the largest palindrome made from the product of two 3-digit
numbers."""

def is_palindrome(number):
    return str(number) == ''.join(reversed(str(number)))


def three_digits():
    return range(100, 999)


max(x*y for x in three_digits() for y in three_digits() if is_palindrome(x*y))
