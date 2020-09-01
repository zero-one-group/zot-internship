#!/usr/bin/env python

num_word_dict={1:len("one"),2:len("two"),3:len("three"),
4:len("four"),5:len("five"), 6:len("six"),7:len("seven"),
8:len("eight"),9:len("nine"),10:len("ten"),11:len("eleven"),
12:len("twelve"),13:len("thirteen"),14:len("fourteen"),
15:len("fifteen"),16:len("sixteen"),17:len("seventeen"),
18:len("eighteen"),19:len("nineteen"),20:len("twenty"),
30:len("thirty"),40:len("fourty"),50:len("fifty"),
60:len("sixty"),70:len("seventy"),80:len("eighty"),
90:len("ninety"),100:len("onehundred"),200:len("twohundred"),
300:len("threehundred"),400:len("fourhundred"),
500:len("fivehundred"),600:len("sixhundred"),700:len("sevenhundred"),
800:len("eighthundred"),900:len("ninehundred"),1000:len("onethousand")}

sum_num = 0
for x in range(1,1001):
    s_sum = sum_num
    if len(str(x)) == 4: 
        sum_num += num_word_dict[1000]
    elif len(str(x)) == 3:
        hundreds = int(str(x)[0]+"00")
        if x % 100 != 0: 
            sum_num += num_word_dict[hundreds]+3
        else: 
            sum_num += num_word_dict[hundreds]
            tens = int(str(x)[1]+"0")
            if tens != 00: 
                sum_num += num_word_dict[tens]
                ones = int(str(x)[2])
            if ones!=0: 
                sum_num += num_word_dict[ones]
            elif len(str(x)) == 2 and str(x)[0]!="1":
                tens = int(str(x)[0]+"0")
                if tens!=00: 
                    sum_num += num_word_dict[tens]
                    ones = int(str(x)[1])
                    if ones!=0: 
                        sum_num += num_word_dict[ones]
                    elif len(str(x))==2 and str(x)[0]=="1":
                        sum_num += num_word_dict[x]
                    else:
                        sum_num += num_word_dict[x] 

print(x, sum_num , sum_num - s_sum)

print(sum_num)
