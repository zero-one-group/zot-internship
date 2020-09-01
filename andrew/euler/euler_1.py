'''Find the sum of all the multiples of 3 or 5 below 1000'''

lower_bound = int(input("Enter the lower boundary of the range: "))
upper_bound = int(input("Enter the upper boundary of the range: "))

multiples = [multiples for multiples 
        in range(lower_bound, upper_bound+1) 
        if multiples % 3 == 0 or multiples % 5 == 0]
total = sum(multiples)
print("Sum =", total)

