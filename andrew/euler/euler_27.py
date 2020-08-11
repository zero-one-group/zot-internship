'''
Find the product of the coefficients, a and b, for the quadratic expression that produces the maximum number of primes for consecutive values of n, starting with n=0.
'''

from math import sqrt
from itertools import product, count, islice

def is_prime(num):
    return num > 1 and all(num % i for i in islice(count(2), int(sqrt(num)-1)))

# As the quadratic formula has to provide us with primes all the way from 0, this means that b must also be a prime
b = []
for num in range(-1000, 1001):
    if is_prime(abs(num)):
        b.append(num)

# All primes except for 2 are odd. When n=1, ans=1+a+b must be odd, so a has to be odd as well.
a = range(-999, 1000, 2)

possibilities = list(product(a, b))
num_of_primes = []
a_seq = []
b_seq = []

for idx in range(len(possibilities)):
    n = 0
    ans = 2
    while is_prime(ans):
        ans = n*n + n*possibilities[idx][0] + possibilities[idx][1]
        n += 1
    num_of_primes.append(n-1)
    a_seq.append(possibilities[idx][0])
    b_seq.append(possibilities[idx][1])

max_seq_idx = num_of_primes.index(max(num_of_primes))
print("a =", a_seq[max_seq_idx])
print("b =", b_seq[max_seq_idx])
print("a x b =", a_seq[max_seq_idx]*b_seq[max_seq_idx])
print("Sequence =", num_of_primes[max_seq_idx])
