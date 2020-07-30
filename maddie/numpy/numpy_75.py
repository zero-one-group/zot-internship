#!/usr/bin/env python
#numpy exercise questions 51 - 75

import numpy as np

#51
#structured array representing position (x, y) and color (r, g, b)
coordinate_colours = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(coordinate_colours)

#52
#convert 32 bit array into an integer 32 bit in place
thirty_two_bit = (np.random.rand(10)*100).astype(np.float32)
integer_32_bit = thirty_two_bit.view(np.int32)
integer_32_bit[:] = thirty_two_bit
print(thirty_two_bit)

#53
#random vector with shape (100, 2) represents coordinates - find point by point distances
rand_vector = np.random.random((100, 2))
X,Y = np.atleast_2d(random_vector[:, 0], random_vector[:, 1])
distances = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(distances)

#54
#generate fake file
fake_file = StringIO('''1, 2, 3, 4, 5
                6,  ,  , 7, 8
                 ,  , 9, 10, 11
''')
#read from file
read_file = np.genfromtxt(fake_file, delimiter=",", dtype=np.int)
print(read_file)

#55
#equivalent of enumerate in numpy
array = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(array):
    print(index, value)
for index in np.ndindex(array.shape):
    print(index, array[index])

#56
#generate generic 2D Gaussian-like array
X, Y = np.meshgrid(np.linspace(-1,1,10)), np.linspace(-1,1,10))
calculate = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
gaussian_array = np.exp(-( (calculate-mu)**2 / (2.0 * sigma**2) ) )
print(gaussian_array)

#57
#randomly place p elements in a 2D array
array_2D = np.arange(10).reshape(5, 2)
print(array_2D)

p = 3
np.put(array_2D, np.random.choice(range(10), p, replace=False),1)
print(array_2D)

#58
#Substract mean of each row of a matrix
matrix = np.random.rand(5, 5)
subtract_mean_from_rows = N - N.mean(axis=1, keepdims=True)
print(matrix)

#59
#sort an array by the nth column 
array = np.random.randint(0, 9,(3, 3))
print(array)
print(array[array[:,1].argsort()])

#60
#tell if 2D array has null columns
array = np.random.randint(0, 9,(3, 3))
print(array)
print((~array.any(axis=0)).any())

#61
#find nearest value from a given value in an array
array = np.random.uniform(0, 1, 10)
value = 0.2
nearest_value = array.flat[np.abs(array - value).argmin()]
print(nearest_value)

#62
#compute sum of 2 arrays with shape (1, 3) and (3, 1) using interator
array = np.arange(3).reshape(1, 3)
array2 = np.arange(3).reshape(3, 1)
compute_sum = np.nditer([array, array2, None])
for m, n, o in compute_sum: o[...] = m + n
print(compute_sum.operands[2])

#63
#create an array class that has a name attribute
class Array(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

test_array = Array(np.arange(10), "upto_10")
print(test_array.name)

#64
#add 1 to each element indexed by a second vector
new_arr = np.arange(10)
print(new_arr)
ind = np.arange(len(new_arr))
print(ind)
np.add.at(new_arr, ind, 1)
print(new_arr)

#65
#accumulate elements of a vector (X) to an array (F) based on an index list (I)
X = [1, 2, 3, 4, 5]
I = [1, 6, 4, 2, 3]
F = np.bincount(I, X)
print(F)

#66
#w,h,3 image of dtype=ubyte compute the number of unique colors
w,h = 10,10
image = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
colours = image[...,0]*256*256 + image[...,1]*256 + image[...,2]
unique_colours = len(np.unique(colours))
print(unique_colours)

#67
#4D array, sum over last two axis at once
array_4D = np.random.randint(0, 5,(5, 6, 5, 6))
sum_last_two_axis = F.sum(axis=(-2, -1))
print(sum_last_two_axis)


#68
#use a 1D vector (D) to compute means of subsets of D using vector S of same size describing subset indices

D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

#69
#Diagonal of a dot product
random_array = np.random.randint(0, 5,(5, 5))
random_array2 = np.random.randint(0, 5,(5, 5))
print(np.diag(np.dot(random_array, random_array2)))

#70
#vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value
vector = np.array([1, 2, 3, 4, 5])
no_zeros = 3
z0 = np.zeros(len(vector) + (len(vector)-1)*(no_zeros))
z0[::z+1] = vector
print(z0)

#71
#multiply array of dimensions (5, 5, 3) by an array with dimensions (5, 5)
array = np.random.randint(0, 10,(5, 5, 3))
print(array)
array_new_dimensions = 2*np.ones((5, 5))
print(array * array_new_dimensions[:, :, None])

#72
#swap two rows of an array
array = np.arange(25).reshape(5, 5)
print(array)
array[[0, 1]] = array[[1, 0]]
print(array)

#73
#find set of unique line segments composing all the triangles from set of 10 triplets describing 10 triangles (with shared vertices)
faces = np.random.randint(0, 100,(10, 3))
shift = np.roll(faces.repeat(2, axis=1), -1, axis=1)
shift = shift.reshape(len(shift)*3,2)
shift = np.sort(shift, axis=1)
unique_line_segments = shift.view( dtype=[('p0', shift.dtype), ('p1', shift.dtype)] )
unique_line_segments = np.unique(unique_line_segments)
print(unique_line_segments)

#74
#array C is a bincount, how to produce an array A such that np.bincount(A) == C
C = np.bincount([1, 1, 2, 3, 4, 4, 6])
A = np.repeat(np.arange(len(C)), C)
print(A)

#75
#compute averages using a sliding window over an array
def moving_average(a, n=3):
    return_cumsum = np.cumsum(a, dtype=float)
    return_cumsum[n:] = return_cumsum[n:] - return_cumsum[:-n]
    return return_cumsum[n - 1:] / n
data = np.arange(20)
print(moving_average(data, n=3))




