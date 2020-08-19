"""Find the maximum total from top to bottom of the triangle below"""

import re
import numpy as np


data_triangle = '''75
95 64
17 47 82
18 35 87 10
20 04 82 47 65
19 01 23 75 03 34
88 02 77 73 07 63 67
99 65 04 28 06 16 70 92
41 41 26 56 83 40 80 70 33
41 48 72 33 47 32 37 16 94 29
53 71 44 65 25 43 91 52 97 51 14
70 11 33 28 77 73 17 78 39 68 17 57
91 71 52 38 17 14 91 43 58 50 27 29 48
63 66 04 68 89 53 67 30 73 16 69 87 40 31
04 62 98 27 23 09 70 98 73 93 38 53 60 04 23'''


def parse_string_to_triangle(string, size):
    np_triangle = np.array(re.split(' |\n', string), dtype=int)
    triangle_shape = np.tril(np.ones(size))
    triangle_shape[triangle_shape > 0] = np_triangle
    return triangle_shape


def maximum_sum(triangle):
    max_sum = triangle
    row = len(max_sum) - 2
    while row >= 0:
        for i in range(len(max_sum[row]) - 1):
            max_sum[row][i] += max(max_sum[row+1][i], max_sum[row+1][i+1])
        row -= 1
    return max_sum[0][0]


data = parse_string_to_triangle(data_triangle, 15)
maximum_sum(data)
