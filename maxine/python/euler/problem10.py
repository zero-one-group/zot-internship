"The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.

Find the sum of all the primes below two million."

def is_prime(n):
    if n>1:
            for d in range (2,n):
                if n % d ==0:
                    return False
            return True
sum(i for i in range(2000000) if is_prime(i))
