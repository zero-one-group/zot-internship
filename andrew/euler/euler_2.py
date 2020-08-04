'''
Consider the terms in the Fibonacci sequence whose values do not exceed four million.
Find the sum of the even valued terms
'''
# initialising sequence
a = [0, 1]

# making fibonacci sequence
while (a[len(a)-1]) < 4000000:
    a.append(a[len(a)-1] + a[len(a)-2])

# delete zero as it is not part of fibonacci sequence
del a[0]

# find even numbers
num = []
print("Even fibonacci numbers lower than 4 million:")
for x in a:
    if x % 2 == 0:
        print(x)
        num.append(x)

# calculate the total
print("\nThe sum is: ")
total = sum(num)
print(total)
