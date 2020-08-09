'''
How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
'''

import datetime

count = 0
for year in range(1901, 2001):
    for month in range(1,13):
        day = datetime.datetime(year, month, 1).weekday()
        if day == 6:
            count += 1

print("The number of sundays that fell on the first of the month during the twentieth century =", count)

