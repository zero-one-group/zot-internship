'''
What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?

'''

num = int(input("What is the smallest number that is evenly divisible \
by all the number from 1 to (enter value) "))

# start counting and go up in 'den' increment to find the number faster
count = num
cond = True
while cond:
    div = []
    for x in range(1, num+1):
        rem = count % x
        div.append(rem)
    if sum(div) == 0:
        print(count)
        cond = False
    else:
        count = count + num
