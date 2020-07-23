"find largest prime factor of 600851475143"
def prime_factors(n):
    for x in range (2,n):
         if n % x ==0:
             n = n/x
             for i in range (2,n):
                 while n %i ==0:
                     n = n/i
                     yield i
                yield x 
max(prime_factors(13195))
