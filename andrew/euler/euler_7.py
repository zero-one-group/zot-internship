'''
By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
What is the 10001st prime number?
'''
import itertools

def prime_eratosthenes(num):
    non_prime_list = []
    for count in range(2, num+1):
        if count not in non_prime_list:
            yield count
            for j in range(count*count, num+1, count):
                non_prime_list.append(j)

# revised
print(next(itertools.islice(prime_eratosthenes(105000), 10000, None)))
