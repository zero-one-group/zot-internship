#!/usr/bin/env python

def sieve(n):
	is_prime = [True]*n
	is_prime[0] = False
	is_prime[1] = False
	for i in xrange(2,int(math.sqrt(n)+1)):
		index = i*2
		while index < n:
			is_prime[index] = False
			index = index+i
	prime = []
	for i in xrange(n):
            if is_prime[i] == True:
                prime.append(i)
        return prime

def divisors(n):
	divs = [1]
	for i in xrange(2,int(math.sqrt(n))+1):
		if n%i == 0:
			divs.extend([i,n/i])
	return list(set(divs))

primes = sieve(10000)

amicable_nums = []

checked = []

for i in xrange(2,10000):
	if i not in primes and i not in checked:
		da = sum(divisors(i))
		db = sum(divisors(da))
		checked.extend([da,db])
		if i == db:
			if da != db:
				amicable_nums.extend([i,da])

print(sum(amicable_nums))
