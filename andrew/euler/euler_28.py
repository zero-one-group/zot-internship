'''
What is the sum of the numbers on the diagonals in a 1001 by 1001 spiral formed in the same way?
'''
import math

def leading_diagonals(size_of_matrix):
    diagonal = 1
    for count in range(size_of_matrix):
        yield diagonal
        diagonal = diagonal + 2*(count + 1)

def other_diagonals(size_of_matrix):
    diagonal = 1
    for multiplier in range(1, math.ceil(size_of_matrix/2)):
        counter = 0
        while counter < 2:
            counter = counter + 1
            add = 4*multiplier
            diagonal = diagonal + add
            yield diagonal

print("Total =", sum(leading_diagonals(1001)) + sum(other_diagonals(1001)))
