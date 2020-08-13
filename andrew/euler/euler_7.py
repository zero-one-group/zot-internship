'''
By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
What is the 10001st prime number?
'''

def is_prime(num):
    '''Check if a number is a prime'''
    for i in range(2, num):
        if (num % i) == 0:
            return False
    else:
        return num


nth_prime = 10001 
prime = []
count = 2
while len(prime) < nth_prime:
    a = is_prime(count)
    if a == False:
        count += 1
    else:
        prime.append(a)
        count += 1

print("prime number =", prime[nth_prime-1])
