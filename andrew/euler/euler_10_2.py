def prime_check2(n):
    #Declare a set - an unordered collection of unique elements
    multiples = set()
    #Iterate through 2 to n
    for i in range(2, n+1):
        #If i has not been eliminated 
        if i not in multiples:
            #Return prime number
            yield(i)
            #Add multiples of the prime in the range to the 'invalid' set
            multiples.update(range(i*i, n+1, i))


a = int(input("Find the sum of all primes below (enter number) "))
prime_numbers = list(prime_check2(a))

# Calculate the sum using a for loop. Can't use 'sum' function as it 'prime_numbers' is a set
sum = 0
for x in prime_numbers:
    sum = x + sum

print("The sum is",sum)
