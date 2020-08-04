'''
The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
Find the sum of all the primes below two million.
'''

def prime_check(num):
    '''Check for prime number'''
    for i in range(2, num):
        if (num % i) == 0:
            num = False
            return num
    else:
        return num

a = int(input("Find the sum of all primes below (enter number) "))
total = 2 #start from 2 as calculation does not include 2
for j in range(3, a, 2):
    prime = prime_check(j)
    if prime != False:
        total = total + prime
print("The sum is", total)
