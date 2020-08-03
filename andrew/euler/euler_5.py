den = int(input("What is the smallest number that is evenly divisible by all the number from 1 to (enter value) "))

denom = []
for i in range(1, den+1):
    denom.append(i)

# start counting and go up in 'den' increment to find the number faster
count = den
while True:
    div = []
    for x in denom:
        rem = count % x
        div.append(rem)
    if sum(div) == 0:
        print(count)
        break
    else:
        count = count + den
