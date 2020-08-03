# It will search numbers between 'num_start' and 'num_end'
num_start = 100
num_end = 999

N = 0
for a in range(num_start, num_end+1):
    for b in range(num_start, a+1):
        x = a * b
        if x > N:
            s = str(a * b)
            if s == s[::-1]:
                N = a * b
                print(a, "x", b, "=", N)


# Note: this can be made faster and clearer by counting top down (999 to 100 instead of 100 to 999)
# For number of digits above 4 (e.g. between 10000 and 99999), it is better to check top down.
# But it's nice to be able to see the process and other palindromic numbers too
# To make it even faster, can start trialing from 900 instead of 100
