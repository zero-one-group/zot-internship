'By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13. What is the 10 001st prime number?'


from itertools import islice


def is_prime(n):
    for d in range (2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True


def prime_generator():
    num = 1
    while True:
        num += 1
        if is_prime(num):
            yield num
            

array = [x for x in islice(prime_generator(), 100001)]
array[10000]

