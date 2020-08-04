'''Find the sum of all the multiples of 3 or 5 below 1000'''

x1 = int(input("Enter the lower boundary of the range: "))
x2 = int(input("Enter the upper boundary of the range: "))

a = list(range(x1, x2+1))

for x in a:
    if x % 3 == 0 or x % 5 == 0:
        print(x)
