def prime_check(num):
    for i in range(2,num):  
        if (num % i) == 0:
            num = False
            return(num)
            break  
    else:  
        return(num)

a = int(input("Find the sum of all primes below (enter number) "))
total = 2 #start from 2 as calculation does not include 2
for i in range(3,a,2):
    num = prime_check(i)
    if num != False:
        total = total + num
print("The sum is",total)
