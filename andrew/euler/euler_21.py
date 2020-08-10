'''
Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide evenly into n).
If d(a) = b and d(b) = a, where a ≠ b, then a and b are an amicable pair and each of a and b are called amicable numbers.

For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.

Evaluate the sum of all the amicable numbers under 10000.
'''

def divisors(num):
    divs = []
    for i in range(1, num):
        if num % i == 0:
            divs.append(i)
    return divs


# Find all sets of numbers that fits the criteria d(a) = b and d(b) = a
d_a = []
d_b = []
for num in range(1, 10000):
    a = sum(divisors(num))
    b = sum(divisors(a))
    if b == num:
        d_a.append(a)
        d_b.append(b)

# Find index of a = b values and repeated values
idx = []
for rows in range(1, len(d_a)):
    if d_a[rows-1] == d_b[rows]:
        idx.append(rows-1)
for rows in range(0, len(d_a)):
    if d_a[rows] == d_b[rows]:
        idx.append(rows)
idx.sort()

# Remove data based on index, leaving amicable pairs
for i in reversed(idx):
    d_a.pop(i)
    d_b.pop(i)

# Calculate sum of all amicable numbers
total = sum(d_a) + sum(d_b)
print("Sum of all amicable numbers =", total)
