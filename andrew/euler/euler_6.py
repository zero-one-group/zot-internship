def sqr_total(num):
    total = 0
    for i in range(1, num+1):
        squared = i ** 2
        total = total + squared
    return total


num = int(input("Enter number to calculate the difference between the total of the squares of natural numbers and the square of the sum: "))

squared_sum = sqr_total(num)
print("squared_sum: ", squared_sum)

total = 0
for a in range(1, num+1):
    total = total + a
sum_squared = total ** 2
print("sum_squared: ", sum_squared)

diff = sum_squared - squared_sum
print("Difference is", diff)
