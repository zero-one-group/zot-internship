'''
Find the difference between the sum of the squares of the first one hundred natural
numbers and the square of the sum.
'''

def sqr_total(value):
    total = sum(i**2 for i in range(1, value+1))
    return total


num = 100 

squared_sum = sqr_total(num)
print("squared_sum: ", squared_sum)

total = 0
for a in range(1, num+1):
    total = total + a
sum_squared = total ** 2
print("sum_squared: ", sum_squared)

diff = sum_squared - squared_sum
print("Difference is", diff)


