'''
How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
'''

import datetime

day = [datetime.datetime(year, month, 1).weekday()
        for year in range(1901, 2001)
        for month in range(1, 13)]

print("The number of sundays that fell on the first of the month during the twentieth century =", day.count(6))


