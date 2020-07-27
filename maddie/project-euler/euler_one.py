#!/usr/bin/env python
TOTAL = 0
for i in range(1, 1000):
    if (i % 3 == 0 or i % 5 == 0):
        TOTAL = TOTAL + i
print(TOTAL)
