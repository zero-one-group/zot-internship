'''
What is the index of the first term in the Fibonacci sequence to contain 1000 digits?
'''

from itertools import takewhile

def fib():
    prev, after = 1, 1
    count = 1
    while True:
        prev, after = after, prev+after
        count += 1
        if len(str(prev)) == 1000:
            break
    yield count


for idx in fib():
    print("Index =", idx)

