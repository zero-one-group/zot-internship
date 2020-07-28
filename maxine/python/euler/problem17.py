"""Number letter counts
Show HTML problem content  
Problem 17
If the numbers 1 to 5 are written out in words: one, two, three, four, five,
then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.

If all the numbers from 1 to 1000 (one thousand) inclusive were written out in
words, how many letters would be used?"""

import inflect
p = inflect.engine()        #library to convert number to words

def letters_in_numbers(num):
    words = p.number_to_words(num).replace('-', ' ').split()
    return sum(len(i) for i in words)
sum(letters_in_numbers(i) for i in range(1000))
