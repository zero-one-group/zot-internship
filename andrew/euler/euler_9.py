import math

def pythagoras(a,b):
    c = math.sqrt((a**2) + (b**2))
    return c


n = 10000
for a in range(1,n+1):
    for b in range(1,n+1):
        c = pythagoras(a,b)
        if c % 1 == 0:
            sum = a+b+c
            if sum == 1000:
                print("a =",a,"; b =",b,"; c =",c,"; sum=",sum)
                exit()
