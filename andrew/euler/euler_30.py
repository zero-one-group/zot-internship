'''
Find the sum of all the numbers that can be written as the sum of fifth powers of their digits.
'''

def power(list, power):
    return [x**power for x in list]


total = 0
for num in range(2, 350000):
    digits = [int(digit) for digit in str(num)]
    answer = power(digits, 5)
    if sum(answer) == num:
        total += num

print("Sum =", total)
