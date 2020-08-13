''' Find the largest prime factor of the number 600851475143 '''

number = 600851475143
prime_factor = []

for divisor in range(2, int(number/2)):
    if number % divisor == 0:
        number = number/divisor
        prime_factor.append(divisor)
    elif number < divisor:
        break

print("Largest prime factor number is", max(prime_factor))


