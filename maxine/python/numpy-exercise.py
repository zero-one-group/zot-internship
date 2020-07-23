"numpy exercises"
import numpy as np          #1
print(np.show_config())
print(np.__version__)       #2
arr = np.zeros(10)                #3
print("%d bytes" % (arr.size * arr.itemsize))   #4 - memory of array
np.info(np.add)             #5 open documentation for np.add

arr1 = np.zeros(10)
arr1[4] = 1
arr1                        #6

arr2 = np.arange(10,50)
arr2                        #7

arr2[::-1]                  #8 reverse arr2

np.arange(9).reshape(3,3)   #9 reshape array

np.nonzero([1,2,0,0,4,0])   #10 indices of non-zeros

np.eye(3)                   #11

np.random.random_sample((3,3))    #12 random array 3x3x3

np.max(np.random.random_sample((10,10)))
np.min(np.random.random_sample((10,10)))    #13 min/max vals

np.mean(np.random.random_sample(30))        #14 mean val

arr3 =np.ones((5,5))
arr3[1:-1, 1:-1] = 0
arr3                                        #15 

np.pad(arr3, (0,0), mode='edge')            #16

0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1                              #17

np.diag((1,2,3,4), k=-1)                    #18

arr4 = np.zeros((8,8))
#odd rows, even cloumns                     #19 
arr4[1::2,::2] = 1
# (even_rows, odd_columns)
arr4[::2,1::2] = 1
arr4

np.unravel_index(100, (6,7,8))            #20

np.tile(arr4, 1)                          #21 make checkboard

arr5 = np.random.randn(5,5)
arr5 = (arr5-np.mean(arr5)/np.std(arr5))
arr5                                        #22 normalize a 5x5 random matrix

np.dtype()      #23 -tbc

arr6 = np.random.randn(5,3)
arr7 = np.random.randn(3,2)
np.dot(arr6, arr7)                  #24 multiply two matrixes

arr8 = np.arange(11) #1-D array examplye
arr8[3:9]*-1                        #25

np.sum(range(5))            #26

#27
#28
#29

np.intersect1d(arr4,arr4)           #30 - common values between two arrays
#31
#32
#33
#34
#35
#36
#37
#38
#39
#40
#41
#42
#43
#44
#45
#46
#47
#48
#49
#50#31

