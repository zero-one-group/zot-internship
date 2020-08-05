'''Find the sum of all the multiples of 3 or 5 below 1000'''

lower_bound= int(input("Enter the lower boundary of the range: "))
upper_bound = int(input("Enter the upper boundary of the range: "))

x = [x for x in range(lower_bound, upper_bound+1) if x % 3 == 0 or x % 5 ==0]
total = sum(x)
print("Sum =", total)

