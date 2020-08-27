'''
Find the sum of the digits in the number 100!
'''

import math

ans = str(math.factorial(100))
numbers = [ans[i:i+1] for i in range(0, len(ans), 1)]
numbers = list(map(int, numbers))
print("Sum of the digits =", sum(numbers))

