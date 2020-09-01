import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


vec1 = np.array([ -1., 4., -9.])
mat1 = np.array([[ 1., 3., 5.], 
                 [7., -9., 2.], 
                 [4., 6., 8. ]])

vec2 = (np.pi/4) * vec1

vec2 = np.cos(vec2)

vec3 = vec1 + 2 * vec2

norm = la.norm(vec3)

vec4 = np.dot(mat1, vec3)

trans = np.transpose(mat1)

det = np.linalg.det(mat1)

trace = np.trace(mat1)

print(min(vec1))

print(np.where(vec1 == min(vec1)))

print(np.amin(mat1))

matrix_a = np.array([[17, 24, 1, 8, 15],
                     [23, 5, 7, 14, 16],
                     [ 4, 6, 13, 20, 22],
                     [10, 12, 19, 21, 3],
                     [11, 18, 25, 2, 9]])

print(np.add.reduce(matrix_a, 0))
print(np.add.reduce(matrix_a, 1))
print(sum(np.diag(matrix_a)))
print(sum(np.diag(np.fliplr(matrix_a))))

random_number = np.random.rand(100)
random_matrix = random_number.reshape(10, 10)

upper_left = random_matrix[:5:1, :5:1]
upper_right = random_matrix[:5:1, 5::]
lower_left = random_matrix[5::, :5:1]
lower_right = random_matrix[5::, 5::]

