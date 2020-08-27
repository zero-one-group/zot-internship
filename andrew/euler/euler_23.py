'''
Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.
'''
import numpy as np

def is_abundant(num):
    total = 0
    for i in range(1, int(num/2 + 1)):
        if num % i == 0:
            total = total + i
    if total > num:
        return True
    else:
        return False

    
abundants = [x for x in range(28124) if is_abundant(x)]

sum_of_two_abundants = [
        abundants[i] + abundants[j]
        for i in range(len(abundants))
        for j in range(i, len(abundants))
        if abundants[i] + abundants[j] < 28123]

sum_of_two_abundants = np.unique(sum_of_two_abundants)

answer = [num for num in range(28123) if num not in sum_of_two_abundants]
print(sum(answer))
