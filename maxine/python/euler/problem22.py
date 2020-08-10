"""What is the total of all the name scores in the file?"""


def parse_txt(txt_file):
    name_file = open(txt_file, "r")
    names = sorted(name_file.read().replace('"','').split(','),key=str)
    return names


def alphabet_value(word):
    return sum((ord(uppercase_letter) - 64) for uppercase_letter in word) 


def scoring(name_list):
    score = 0
    for position, name in enumerate(name_list):
        score += (position + 1) * alphabet_value(name)
    return score


names = parse_txt("p022_names.txt")
scoring(names)
