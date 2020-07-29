"""The following iterative sequence is defined for the set of positive integers:

    n → n/2 (n is even)
    n → 3n + 1 (n is odd)

    Using the rule above and starting with 13, we generate the following
    sequence:

        13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
        It can be seen that this sequence (starting at 13 and finishing at 1)
        contains 10 terms. Although it has not been proved yet (Collatz
                                                                Problem), it is
        thought that all starting numbers finish at 1.

        Which starting number, under one million, produces the longest chain?

        NOTE: Once the chain starts the terms are allowed to go above one
        million"""

import itertools
def number_rule(n):
    if n % 2 == 0:
        return n/2
    else:
        return 3*n + 1

def chain_of_iterative_seq(n):
    x = 1
    while n > 1:
        n = number_rule(n)
        x += 1
    return x

print ("the longest chain are (sequence, starting no.)")
max((chain_of_iterative_seq(i),i) for i in range(1000000))
