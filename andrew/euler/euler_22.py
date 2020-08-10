'''
Using names.txt (right click and 'Save Link/Target As...'), a 46K text file containing over five-thousand first names, begin by sorting it into alphabetical order. Then working out the alphabetical value for each name, multiply this value by its alphabetical position in the list to obtain a name score.

For example, when the list is sorted into alphabetical order, COLIN, which is worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain a score of 938 × 53 = 49714.

What is the total of all the name scores in the file?
'''


file = open("names.txt", "r")
names = file.read()
file.close()

# Split string to individual names
names = names.replace('"','')
names = names.split(',')

# Sort the names alphabetically
names.sort()

# Convert character in names to numbers, add them, and save them into a list
alphabet_values = []
for name in names:
    tot = 0
    for alphabet in range(0, len(name)):
        num = ord(name[alphabet]) - 64
        tot = tot + num
    alphabet_values.append(tot)

# Multiply by the names' index
for idx in range(0, len(names)):
    alphabet_values[idx] = alphabet_values[idx] * (idx+1)

print("Name scores =", sum(alphabet_values))
