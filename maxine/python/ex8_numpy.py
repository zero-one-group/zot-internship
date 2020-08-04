import numpy as np


vec1 = np.array([ -1., 4., -9. ])
mat1 = np.array([[ 1., 3., 5. ], [7., -9., 2.], [4., 6., 8. ]])

vec2 = (np.pi/4) * vec1
vec2
vec2 = np.cos(vec2)
vec2
vec3 = vec1 + 2*vec2
vec3
la.norm(vec3)
vec4 = vec3 * mat1
vec4

mat1_transpose = mat1.transpose()
mat1_transpose

mat1_determinant = np.linalg.det(vec4)
mat1_determinant

mat1_trace = np.ndarray.trace(mat1)
mat1_trace

np.min(vec1)        #smallest element in vec1

np.where(vec1 == np.min(vec1))      #location of smallest element

mat1.min()

a=np.array([[17, 24, 1, 8, 15],
           [23, 5, 7, 14, 16],
           [ 4, 6, 13, 20, 22 ],
           [10, 12, 19, 21, 3],
           [11, 18, 25, 2, 9]])

np.sum(a, axis=0)
np.sum(a, axis=1)
np.sum(np.diag(a))
np.sum(np.fliplr(a))

m=np.random.randn(10,10)
m

mul = m[:5, :5]
mul

mur = m[5:, :5]
mur

mll = m[:5, 5:]
mll

mlr = m[5:, 5:]
mlr
