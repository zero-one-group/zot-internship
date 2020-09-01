#!/usr/bin/env python

import calendar

calendar.setfirstweekday(6)

def sundays(year):
    counter = 0
    for month in range(1, 13):
        cal = calendar.monthcalendar(year, month)
        if cal[0][0]:
            counter += 1

    return counter

total = 0

for i in range(1901, 2001):
    total += sundays(i)

print(total)
