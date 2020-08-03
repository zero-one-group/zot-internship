# Enter number from input
a = int(input("Enter number to find its largest prime factor: "))

prime_factor = []

print("Prime factors:")
# find a number that is divisible to the input, use it to repeat the process until left with 1
for x in range(2, a):
    if a % x == 0:
        a = a/x
        print(x)
#        print(a)
        prime_factor.append(x)
    elif a < x:
        break

# select the largest prime factor
ans = max(prime_factor)
print("Largest prime factor number is", ans)
