'''
The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
Find the sum of all the primes below two million.
'''

def prime_check2(value):
    multiples = set()
    for i in range(2, value+1):
        if i not in multiples:
            yield i
            multiples.update(range(i*i, value+1, i))


input_int = int(input("Find the sum of all primes below (enter number) "))

total = 0
total = [prime+total for prime in list(prime_check2(input_int))]

print("The sum is", sum(total))
