#!/usr/bin/env python

def pythag_triplet():
    for a in range(1, 1000):
        for b in range(a + 1, 1000 - a):
            c = 1000 - a - b

            if a**2 + b**2 == c**2:
                return a, b, c

pythag_triplet()
print("a:", a, "b:", b, "c:", c)

