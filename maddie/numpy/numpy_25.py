#!/usr/bin/env python
import numpy as np

print(np.__version__) #numpy version and configuration
print(np.show_config())

null_vector = np.zeros(10) #create null vector size 10
print(null_vector)
print("%d bytes" % (x.size * x.itemsize)) #memory size of array x

new_array = np.linspace(10, 49, num=50) #vector with values from 10 to 49
print(new_array)
new_array_flip = np.flip(new_array, 0) #reverse vector
print(new_array_flip)

new_array = np.arange(0, 9).reshape(3, 3) #create matrix with 3 x 3 size values 0-8
print(new_array)

new_array = np.array([1, 2, 0, 0, 4, 0])
print(np.nonzero(new_array)) #print all indices of nonzero values

array_2D = np.identity(3) #create 3x3 identity matrix
print(array_2D)

random_array = np.random.random((3, 3, 3)) #create 3x3x3 array with random values
print(random_array)

new_random = np.random.random((10, 10)) #create 10x10 array with random values
print(np.amin(new_random)) #print min and max values
print(np.amax(new_random))

random_array = np.random.random(30) #create random array of size 30 and print mean
print(np.mean(random_array)) 

ones_array = np.ones((5, 5)) #create array with 1s as border to 0s
ones_array[1:-1, 1:-1] = 0
print(ones_array)

border_zeros = np.pad(h, pad_width = 1, mode = 'constant', constant_values = 0) #create border of 0s
print(border_zeros)

0 * np.nan # check result of the following code (provided by the exercise)
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1

new_matrix = np.diag([1, 2, 3, 4, 5]) #create a 5x5 matrix with values 1,2,3,4 just below the diagonal
print(new_matrix)

def printcheckboard(p):
    z = np.zeros((p, p), dtype=int)
    z[1::2, ::2] = 1
    z[::2, 1::2] = 1
    for i in range(p):
        for j in range(p):
            print(z[i][j], end=" ")
        print()

p = 8
printcheckboard(p)

print(np.unravel_index(100, (6, 7, 8))) #check index of 100th element in a 6,7,8 shape array

array = np.array([[0, 1], [1, 0]])
checkerboard = np.tile(array,(4, 4)) #checkerboard from tile function
print(checkerboard)

random_matrix = np.random.random((5, 5))
random_matrixmax, random_matrixmin = random_matrix.max(), random_matrix.min() #normalize the random matrix
random_matrix = (random_matrix-random_matrixmin)/(random_matrixmax-random_matrixmin)
print(random_matrix)

#custom dtype that describes a colour as four unsigned bytes (RGBA)
colour = np.dtype([("r", np.ubyte, 1),
                   ("g", np.ubyte, 1),
                   ("b", np.ubyte, 1),
                   ("a", np.ubyte, 1)])
print(colour)

multiply_matrix = np.dot(np.ones((5, 3)), np.ones((3, 2))) #multiply a 5x3 matrix by a 3x2 matrix
print(multiply_matrix)

negative_numbers = np.arange(11)
negative_numbers[(3 < negative_numbers) & (negative_numbers < 8)] *= -1 #elements between 3-8 are negative numbers
print(negative_numbers)
