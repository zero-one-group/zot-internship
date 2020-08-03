#!/usr/bin/env python

def pythag_triplet():
    for triplet_one in range(1, 1000):
        for triplet_two in range(triplet_one + 1, 1000 - triplet_one):
            triplet_three = 1000 - triplet_one - triplet_two

            if triplet_one**2 + triplet_two**2 == triplet_three**2:
                return triplet_one, triplet_two, triplet_three

pythag_triplet()
print("a:", triplet_one, "b:", triplet_two, "c:", triplet_three)

