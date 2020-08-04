#!/usr/bin/env python
two = 0
three = 0
five = 0
seven = 0
for i in range(2, 20):
    if i % 2 == 0:
        two = two + 1
    elif i % 3 == 0:
        three = three + 1
    elif i % 5 == 0:
        five = five + 1
    elif i % 7 == 0:
        seven = seven + 1
    else:
        break

n = (2**two) * (3**three) * (5**five) * (7**seven) * 11 * 13 * 17 * 19
print(n)
