"""What is the value of the first triangle number to have over five hundred divisors?"""

from functools import reduce


def sequence_num_to_value(num):
    return int(num*(num + 1) / 2)


def factors(num):
    seq = sequence_num_to_value(num)
    return set(reduce(list.__add__, ([x, int(seq/x)]
                                     for x in range(1, int(seq**0.5) + 1) if not seq % x)))


def first_sequence_with_divisor(divisor):
    count = 1
    while len(list(factors(count))) <= divisor:
        count += 1
    return sequence_num_to_value(count)


first_sequence_with_divisor(500)
