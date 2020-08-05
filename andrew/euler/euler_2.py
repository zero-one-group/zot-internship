'''
Consider the terms in the Fibonacci sequence whose values do not exceed four million.
Find the sum of the even valued terms
'''


prev, after = 0, 1
total = 0
while True:
    prev, after = after, prev + after
    if after >= 4000000:
        break
    if after % 2 == 0:
        total += after

print("Sum of the even valued terms of Fibonacci sequence below four million:", total)
