'''
A palindromic number reads the same both ways.
The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.
Find the largest palindrome made from the product of two 3-digit numbers.
'''

def palindrome_finder(num_start, num_end):
    '''Receives start and end numbers to check from palindromic numbers'''
    ans = 0
    for first_prod in range(num_start, num_end+1):
        for second_prod in range(num_start, first_prod+1):
            product = first_prod * second_prod
            if product > ans:
                res_str = str(first_prod * second_prod)
                if res_str == res_str[::-1]:
                    ans = first_prod * second_prod
                    print(first_prod, "x", second_prod, "=", ans)


palindrome_finder(100, 999)
# Note: this can be made faster and clearer by counting top down (999 to 100 instead of 100 to 999)
# For number of digits above 4 (e.g. between 10000 and 99999), it is better to check top down.
# But it's nice to be able to see the process and other palindromic numbers too
# To make it even faster, can start trialing from 900 instead of 100
