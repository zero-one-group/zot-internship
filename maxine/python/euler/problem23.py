"""Find the sum of all the positive integers which cannot be written as the
sum of two abundant numbers."""


def is_abundant(num):
    if num > sum(i for i in range(1, num) if num % i == 0):
        return True


def cannot_be_written_as_two_abundant(num):
    return (i for i in range(num) if i is not is_abundant(int(i/2)))


sum(cannot_be_written_as_two_abundant(28123))
