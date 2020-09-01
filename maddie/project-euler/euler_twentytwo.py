#!/usr/bin/env python

def name_score(name):
	letters = list(name)
        letters = [ord(x)-64 for x in letters]
        return sum(letters)

with open('names.txt') as f:
    names = f.read()

names = names.strip().split(',')

names = [element[1:-1] for element in names]

names.sort()

scores = 0

for i in range(len(names)):
    scores += name_score(names[i])*(i+1)

print scores

