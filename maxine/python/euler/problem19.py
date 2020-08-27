"""How many Sundays fell on the first of the month during the twentieth century
(1 Jan 1901 to 31 Dec 2000)?"""

from datetime import date


def sundays(day, start_year, end_year):
    sunday = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if date(year, month, day).weekday() == 6:
                sunday += 1
    return sunday


print("There are %d sundays on the first month in the twentieth century" %
      sundays(1, 1901, 2000))
