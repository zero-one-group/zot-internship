'''100 numpy exercises'''

# 1. Import the numpy package under the name np
import numpy as np

# 2. Print the numpy verstion and the congfiguration
print(np.__version__)
print(np.show_config())

# 3. Create a null vector of size 10
null_vector = np.zeros(10)

# 4. How to find the memory size of any array
memory = "%d bytes" % (null_vector.size * null_vector.itemsize)
print(memory)

# 5. How to get the documentation of the numpy add function from the command line?
np.info(np.add)

# 6. Create a null vector of size 10 but the fifth value which is 1
null_vector = np.zeros(10)
null_vector[4] = 1
print(null_vector)

# 7. Create a vector with values ranging from 10 to 49
vector = np.arange(10, 50)
print(vector)

# 8. Reverse a vector (fifth element becomes last)
vector = np.arange(5)
vector = vector[::-1]
print(vector)

# 9. Create 3x3 matrix with values ranging from 0 to 8
matrix = np.arange(0, 9).reshape(3, 3)
print(matrix)

# 10. Find indices of non-zero elements form [1,2,0,0,4,0]
arr = np.array([1, 2, 0, 0, 4, 0])
index = np.where(arr != 0)
print(index)

# 11. Create a 3x3 identity matrix
iden_matrix = np.identity(3)
print(iden_matrix)

# 12. Create a 3x3x3 array with random values
rand_matrix = np.random.rand(3, 3, 3)
print(rand_matrix)

# 13. Create a 10x10 array with random values and find the minimum and maximum values
rand_matrix = np.random.rand(10,10)
min_value = np.min(rand_matrix)
max_value = np.max(rand_matrix)
print("Minimum value =", min_value, "; Maximum value =", max_value)

# 14. Create a random vector of size 30 and find the mean value
rand_matrix = np.random.rand(30)
mean_value = np.mean(rand_matrix)
print("Mean =", mean_value)

# 15. Create a 2d array with 1 on the border and 0 inside
matrix = np.ones((10,10))
matrix[1:-1, 1:-1] = 0
print(matrix)

# 16. How to add a border (filled with 0's) around an existing array?
matrix = np.ones((8,10))
matrix[0, :] = 0
matrix[:, 0] = 0
matrix[-1, :] = 0
matrix[:, -1] = 0
print(matrix)

# 17. What is the result of the following expression?
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)

# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
matrix = np.diag(np.arange(1, 5), k = -1)
print(matrix)

# 19. Create a 8x8 matrix and fill it with a checkerboard pattern
matrix = np.zeros((8, 8))
matrix[0::2, 0::2] = 1
matrix[1::2, 1::2] = 1
print(matrix)

# 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
#print(np.unravel_index(99, (6,7,8)))
matrix = np.zeros((6, 7, 8))
matrix = np.arange(0, matrix.size).reshape(6, 7, 8)
index = np.where(matrix == 99)
print(index)

# 21. Create a checkerboard 8x8 matrix using the tile function
array = np.array([[0, 1], [1, 0]])
checkerboard = np.tile(array, (4, 4))
print(checkerboard)

# 22. Normalise a 5x5 random matrix
rand_matrix = np.random.rand(5, 5)
max_value = rand_matrix.max()
min_value = rand_matrix.min()
norm_matrix = (rand_matrix - min_value)/(max_value - min_value)
print(norm_matrix)

# 23. Create a custom dtype that describes a colour as four unsigned bytes (RGBA)
colour = np.dtype([("r", np.ubyte, 1),
                   ("g", np.ubyte, 1),
                   ("b", np.ubyte, 1),
                   ("a", np.ubyte, 1)])
print(colour)

# 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
A = np.random.rand(5, 3)
B = np.random.rand(3, 2)
product = A @ B
print(product)

# 25. Given a 1D array, negate all elements which are between 3 and 8, in place.
array = np.arange(100)
array[(array >= 3) & (array <= 8)] *= -1
print(array)

# 26. What is the output of the following script?
print(sum(range(5),-1))
#from numpy import *
print(sum(range(5),-1))

# 27. Consider an integer vector Z, which of these expressions are legal?
Z = np.arange(5)
print(Z)
print(Z**Z)
print(2 << Z >> 2)
print(Z <- Z)
print(1j*Z)
print(Z/1/1)

# 28. What are the result of the following expressions?
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))

# 29. How to round away from zero a float array?
vector = np.random.uniform(-10,10,10)
print(vector)
print(np.where(vector>0, np.ceil(vector), np.floor(vector)))

# 30. How to find common values between two arrays?
vector_1 = np.random.randint(0,10,20)
vector_2 = np.random.randint(0,10,20)
print(vector_1)
print(vector_2)
print(np.intersect1d(vector_1, vector_2))

# 31. How to ignore all numpy warnings (not recommended)?
with np.errstate(all = "ignore"):
    np.arange(3) / 0

# 32. Is the following expressions true?
with np.errstate(all = "ignore"):
    np.sqrt(-1) == np.emath.sqrt(-1) # No, np.sqrt(-1) returns error, np.emath.sqrt returns 1j

# 33. How to get the dates of yesterday, today and tomorrow?
print("Yesterday: ", np.datetime64('today') - np.timedelta64(1))
print("Today: ", np.datetime64('today'))
print("Tomorrow: ", np.datetime64('today') + np.timedelta64(1))

# 34. How to get all the dates corresponding to the month of July 2016?
date = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(date)

# 35. How to compute ((A+B)*(-A/2)) in place (without copy)?
A = np.random.randint(0, 10, 10)
B = np.random.randint(0, 10, 10)
print(A)
print(B)
C = ((A-B)*(-A/2))
print(C)

# 36. Extract the integer part of a random array of positive numbers using 4 different methods?
Z = np.random.rand(10)*10
print(Z)
print(Z - Z%1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))

# 37. Create a 5x5 matrix with row values ranging from 0 to 4
Z = np.zeros((5, 5))
Z = np.arange(5) + Z
print(Z)

# 38. Consider a generator function that generates 10 integers and use it to build an array
def generator():
    for x in range(10):
        yield x
Z = np.fromiter(generator(), dtype=float)
print(Z)

# 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded
vector = np.linspace(0, 1, 11, endpoint = False)
print(vector[1:])

# 40. Create a random vector of size 10 and sort it
vector = np.random.rand(10)
vector.sort()
print(vector)

# 41. How to sum a small array faster than np.sum?
arr = np.arange(10)
print(np.add.reduce(arr))

# 42. Consider two random array A and B, check if they are equal
A = np.arange(5)
B = np.arange(5)
equal = np.array_equal(A, B)
print(equal)

# 43. Make an array immutable (read only)
arr = np.arange(10)
arr.flags.writeable = False
#arr[0] = 1 #returns error

# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
coor = np.random.random((10,2))
x, y = coor[:,0], coor[:,1]
cartesian = np.sqrt(x**2 + y**2)
polar = np.arctan(y, x)
print("Cartesian: ", cartesian)
print("Polar: ", polar)

# 45. Create a random vector of size 10 and replace the maximum value by 0
vector = np.random.random(10)
vector[vector.argmax()] = 0
print(vector)

# 46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area
structured = np.zeros((10,10), [('x',float),('y',float)])
structured['x'], structured['y'] = np.meshgrid(x, y)
print("structured: \n", structured)

# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
arr_1 = np.arange(3)
arr_2 = np.random.random(3) * 10
C = 1 / np.subtract.outer(arr_1, arr_2)
print("Cauchy matrix =\n", C)

# 48. Print the minimum and maximum representable value for each numpy scalar type
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

# 49. How to print all the values of an array?
arr = np.ones((101, 101))
print("shortened: \n", arr)
np.set_printoptions(threshold = np.inf)
print("show all: \n", arr)

# 50. How to find the closest value (to a given scalar) in a vector?
data = np.arange(100)
print('data: \n', data)
to_find = 15.3
index = (np.abs(data - to_find)).argmin()
print(data[index])
