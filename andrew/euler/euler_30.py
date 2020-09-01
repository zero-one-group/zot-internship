'''
Find the sum of all the numbers that can be written as the sum of fifth powers of their digits.
'''

def power(list, power):
    return [x**power for x in list]

total = [num for num in range(2, 350000) 
        if sum(power([int(digit) for digit in str(num)], 5)) == num]

print("Sum =", sum(total))
