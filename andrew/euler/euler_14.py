'''
The following iterative sequence is defined for the set of positive integers:

n → n/2 (n is even)
n → 3n + 1 (n is odd)

Using the rule above and starting with 13, we generate the following sequence:

13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.

Which starting number, under one million, produces the longest chain?
'''

def collatz(num):
    count = 1
    while num != 1:
        if num % 2 == 0:
            num = num/2
            count = count + 1
        else:
            num = 3*num + 1
            count = count + 1
    return count


n = 1000000
chain = [collatz(i) for i in range(1, n + 1)]

index = chain.index(max(chain))
print("Starting point =", index + 1)
print("Number of chain =", chain[index])
