# Enter number from input
a = int(input("Enter number to find its largest prime factor: "))

# Check top down to make it faster
for x in range(a, 1, -1):
    if a % x == 0:
        factor = x
        print(factor)
        for i in range(2, factor):
            if (factor % i) == 0:
                break
        else:
            ans = factor
            break

print("Largest prime factor number is", ans)
