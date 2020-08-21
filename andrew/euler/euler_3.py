''' Find the largest prime factor of the number 600851475143 '''

def is_prime(num):
    for i in range(2, num):
        if num % i == 0:
            return False
    else:
        return num


number = 600851475143
prime_factor = []

for divisor in range(2, int(number/2)):
    if is_prime(divisor):
        if number % divisor == 0:
            number = number/divisor
            print(divisor)
            prime_factor.append(divisor)
        elif number < divisor:
            break

print("Largest prime factor number is", max(prime_factor))


