"""2520 is the smallest number that can be divided by each of the numbers from
1 to 10 without any remainder.
What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?"""


def greatest_common_denominator(num1, num2):
    while num2:
        num1, num2 = num2, num1 % num2
    return num1


def least_common_multiple(num1, num2):
    return (num1 * num2) / greatest_common_denominator(num1, num2)


def smallest_divisible_to_n(num):
    low_least_common_multiple = least_common_multiple(11, 12)
    for n in range(num - 9, num):
        low_least_common_multiple = least_common_multiple(n, low_least_common_multiple)
    return low_least_common_multiple


smallest_divisible_to_n(20)
