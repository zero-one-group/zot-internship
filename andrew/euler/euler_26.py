'''
Find the value of d < 1000 for which 1/d contains the longest recurring cycle in its decimal fraction
'''

max_len = 0
max_d = 1

for d in range(1, 1000):
    quotient = {0: 0}
    cur_value = 1
    len_recur = 0

    while cur_value not in quotient:
        len_recur += 1
        quotient[cur_value] = len_recur
        cur_value = (cur_value % d) * 10
    if cur_value == 0:
        continue

    len_recur -= quotient[cur_value]

    if len_recur > max_len:
        max_len, max_d = len_recur, d
    
print(max_d)

