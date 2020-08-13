'''
2^15 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
What is the sum of the digits of the number 2^1000?
'''

# >>> Revised
power = 1000
answer = str(2**power)
total = sum(map(int, answer))
print(total)

# Original
power = 1000
answer = str(2**power)
# No need, str is an iterable
numbers = [answer[i:i+1] for i in range(0, len(answer), 1)]
# No need, list, leave it lazy, because the next operation is sum
numbers = list(map(int, numbers))
total = sum(numbers)
print(total)
