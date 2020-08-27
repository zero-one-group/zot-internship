'''
Using names.txt (right click and 'Save Link/Target As...'), a 46K text file containing over five-thousand first names, begin by sorting it into alphabetical order. Then working out the alphabetical value for each name, multiply this value by its alphabetical position in the list to obtain a name score.

For example, when the list is sorted into alphabetical order, COLIN, which is worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain a score of 938 × 53 = 49714.

What is the total of all the name scores in the file?
'''

import numpy as np

def length_of_names(name):
    num = [ord(name[alphabet]) - 64 for alphabet in range(0, len(name))]
    return sum(num)

def read_data(filename):
    with open(filename, "r") as file:
        names = file.read()
        names = names.replace('"','').split(',')
        names.sort()
    return names


names = read_data("names.txt")
alphabet_values = [length_of_names(name) for name in names]

name_scores = np.dot(alphabet_values, list(range(1, len(names)+1)))
print("Name scores =", name_scores)
