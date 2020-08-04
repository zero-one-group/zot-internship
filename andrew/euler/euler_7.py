'''
By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
What is the 10 001st prime number?
'''

def prime_check(num):
    '''Check if a number is a prime'''
    for i in range(2, num):
        if (num % i) == 0:
            num = False
            return num
    else:
        return num


nth_prime = int(input("Which prime number do you want? "))
prime = []
count = 2
while len(prime) < nth_prime:
    a = prime_check(count)
    if a == False:
        count = count + 1
    else:
        prime.append(a)
        count = count + 1

print("prime number =", prime[nth_prime-1])
