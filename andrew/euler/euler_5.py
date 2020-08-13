'''
What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?

'''

num = 20
count = num
while True:
    div = [count % denom for denom in range(1, num+1)]
    if sum(div) == 0:
        print(count)
        break
    else:
        count += num

