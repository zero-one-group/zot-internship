#!/usr/bin/env python
Fibonacci1 = 1
Fibonacci2 = 2
a = 2
while True:
    Fibonacci1 = Fibonacci1 + Fibonacci2
    if Fibonacci1 % 2 == 0:
        if a + Fibonacci1 > 4000000:
            break
        a = a + Fibonacci1
    Fibonacci2 = Fibonacci1 + Fibonacci2
    if Fibonacci2 % 2 == 0:
        if a + Fibonacci2 > 4000000:
            break
        a = a + Fibonacci2
print(a)
