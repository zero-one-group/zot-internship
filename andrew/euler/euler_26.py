'''
Find the value of d < 1000 for which 1/d contains the longest recurring cycle in its decimal fraction
'''

def hand_division(numerator, quotient, recurring_length):
    while numerator not in quotient:
        recurring_length += 1
        quotient[numerator] = recurring_length
        numerator = (numerator % divisors) * 10
    return (numerator, quotient, recurring_length)


max_length = 0
longest_recurring_divisors = 1

for divisors in range(1, 1000):
    numerator, quotient, recurring_length = 1, {0: 0}, 0
    numerator, quotient, recurring_length = hand_division(numerator, quotient, recurring_length)
    
    if numerator == 0:
        continue

    recurring_length -= quotient[numerator]
    if recurring_length > max_length:
        max_length, longest_recurring_divisors = recurring_length, divisors
    
print(longest_recurring_divisors)

