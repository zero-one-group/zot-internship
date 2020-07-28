"""215 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26. What is the sum of the digits of the number 21000?"""
def two_power_nth(n):
    return 2**(n)
sum(int(i) for i in str(two_power_nth(1000)))
