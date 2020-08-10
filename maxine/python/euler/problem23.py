"""Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers."""


def is_abundant(x):
    if x > sum(i for i in range (1, x) if x % i == 0):
        return True
    else:
        return False
        

def cannot_be_written_as_two_abundant(x):
    return (i for i in range(x) if i is not is_abundant(int(i/2)))


sum(cannot_be_written_as_two_abundant(28123))

