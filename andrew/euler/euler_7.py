def prime_check(num):
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
