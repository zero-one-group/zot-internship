'''
A palindromic number reads the same both ways.
The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.
Find the largest palindrome made from the product of two 3-digit numbers.
'''

def is_palindrome(res_str):
    return res_str == res_str[::-1]


palindromic_number = [
    (first_prod * second_prod)
    for first_prod in range(100, 1000)
    for second_prod in range(100, first_prod + 1)
    if is_palindrome(str(first_prod * second_prod))
]

print(max(palindromic_number))

