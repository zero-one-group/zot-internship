#!/usr/bin/env python

print(sum(range(5),-1))
import numpy as np
#from numpy import * this imports all functions into the namespace not recommended because of overlap
#print(sum(range(5),-1)) this is actually np.sum second parameter axis (dimension)

#np.array(0) / np.array(0) doesn't work because of zero divide
#np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)

#Let float array principle 0
data = np.random.uniform(-10, +10, 10)
print('data: \n', data)
print(u'After processing: \n', np.copysign(np.ceil(np.abs(data)), data))

#Find the same element of two arrays
data1 = np.arange(25).reshape(5,5)
data2 = np.arange(2, 18).reshape(4, 4)

np.intersect1d(data1, data2)

#Ignore all warnings of numpy(not recommended)
#default = np.seterr(all = 'ignore')
#np.ones(1) / 0
#_ = np.seterr(**default)

#Dates
print('today: ', np.datetime64('today', 'D'))
print('yesterday: ', np.datetime64('today', 'D') - np.timedelta64(1, 'D'))
print('tomorrow: ', np.datetime64('today', 'D') + np.timedelta64(1, 'D'))

#Get all dates in July
np.arange('2020-07', '2020-08', dtype = 'datetime64[D]')

#Calculated in place, cannot be copied
array = np.ones(6, dtype = int).reshape(2, 3)
array_reshape = np.ones(6, dtype = int).reshape(2, 3)*2
array_reshape = array + array_reshape 
array = (-array)/2
array*array_reshape

#Go to the integer part of the array
data = np.random.uniform(0, 10, 10)
print('data: \n', data)
print(u'method one: ', data-data%1)
print(u'method two: ', np.ceil(data)-1)
print(u'method three: ', np.floor(data))
print(u'method four: ', np.trunc(data))
print(u'method 5: ', data.astype(int))

#5x5 matrix with row values ranging from 0-4
data = np.zeros((5, 5), dtype = int)
data += np.arange(5)
print(u'method one: \ n', data)

data = np.tile(np.arange(5, dtype = int), [5, 1])
print(u'method two: \ n', data)

#Generator function that generates 10 integers and then build array
def generate_ten_integers():
    for i in range(10):
        yield i 
data = np.fromiter(generate_ten_integers(), dtype = float, count = -1)
print('data; \n', data)

#create vector size 10 values from 0-1
data = np.linspace(0, 1, 11, endpoint = False)[1:]
data

#create random vector and sort it
random_vector = np.random.random(10)
print('data: \n', random_vector)
random_vector.sort()
print('after sort: \n', random_vector)

#use reduce, quicker than merge and sum
data = np.arange(100000)
%time print('np.sum: ', np.sum(data))
%time print('np.add.reduce: ', np.add.reduce(data))

#check if two arrays are equal
array1 = np.array([1, 2, 3])
array2 = np.array([1, 2, 3])
print(u'Method one (no difference found):', False in np.equal(array1, array2))
print(u'Method Two: ', np.array_equal(array1, array2))

#two arrays are not equal
array1 = np.array([1, 2, 3])
array2 = np.array([1, 2, 4])
print(u'Method one (discover the difference):', False in np.equal(array1, array2))
print(u'Method Two: ', np.array_equal(array1, array2))

#create read-only array
data = np.zeros(10)
data.flags.writeable = False

#matrix representing cartesian coordinates convert to polar coordinates
matrix = np.random.random((10, 2))
X, Y = Z[:, 0], matrix[:, 1]
cartesian_coordinates = np.sqrt(X**2 + Y**2)
polar_coordinates = np.arctan2(Y, X)
print(cartesian_coordinates)
print(polar_coordinates)

#create random vector of size 10 replace max value with 0
data = np.random.random(10)
print('data: \n', data)
data[data.argmax()] = 0
print('data: \n', data)

#create structured array with x, y coordinates
data = np.zeros((5, 5), dtype = [('x', float), ('y', float)])
#meshgrid
data['x'], data['y'] = np.meshgrid(np.linspace(0, 1, 5))
print('data: \n', data)

#construct Cauchy matrix C
data1 = np.arange(10)
data2 = data1 * 5
print('data1: ', data1)
print('data2: ', data2)

cauchy_matrix = 1 / (data1 - data2)
print('result: \n', cauchy_matrix)

#print min and max representable value for each numpy scalar type
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

#print all values of array
data = np.arange(100)
np.set_printoptions(threshold = np.inf)
data

#find the closest value for a given scalar
data = np.random.random(100)
print('data: \n', data)
#print('result: ', )
scalar = 0.2
data[(np.abs(data-scalar)).argmin()]
