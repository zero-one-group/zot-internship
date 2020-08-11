'''
What is the sum of the numbers on the diagonals in a 1001 by 1001 spiral formed in the same way?
'''
import math

size_of_matrix = 1001

# Find the sum of leading diagonal (1 included)
diagonal_1 = 1
total_1 = 1
count_1 = 1
while count_1 < size_of_matrix:
    diagonal_1 += 2*count_1
    total_1 += diagonal_1
    count_1 += 1

# Find the sum of the other diagonal (1 excluded)
diagonal_2 = 1
total_2 = 0
for multiplier in range(1, math.ceil(size_of_matrix/2)):
    counter = 0
    while counter < 2:
        counter += 1
        add = 4*multiplier
        diagonal_2 += add
        total_2 += diagonal_2

print("Total =", total_1 + total_2)


