'Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 3025−385=2640. Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.'

def sum_of_squares(n):
    return(sum(n ** 2 for n in range(1, n + 1)))


def square_of_sum(n): 
   return((sum(n for n in range (1, n + 1))) ** 2)


square_of_sum(23) - sum_of_squares(23)
