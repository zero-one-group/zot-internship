#!/usr/bin/env python
sum([
    x
    for x in range(1, 1001)
    if x % 3 == 0 or x % 5 == 0
])
