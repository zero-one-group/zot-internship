'''
The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
Find the sum of all the primes below two million.
'''

def prime_check2(value):
    '''Check for prime numbers'''
    #Declare a set - an unordered collection of unique elements
    multiples = set()
    #Iterate through 2 to n
    for i in range(2, value+1):
        #If i has not been eliminated
        if i not in multiples:
            #Return prime number
            yield i
            #Add multiples of the prime in the range to the 'invalid' set
            multiples.update(range(i*i, value+1, i))


a = int(input("Find the sum of all primes below (enter number) "))
prime_numbers = list(prime_check2(a))

# Calculate the sum using a for loop. Can't use 'sum' function as it 'prime_numbers' is a set
total = 0
for x in prime_numbers:
    total = x + total

print("The sum is", total)
