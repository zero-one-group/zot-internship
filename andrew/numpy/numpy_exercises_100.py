import numpy as np

# 51. Create a structured array representing a position (x,y) and a color (r,g,b)
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)

# 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances
vector = np.random.random((100,2))
import scipy
import scipy.spatial
distance = scipy.spatial.distance.cdist(vector, vector)
print(distance)

# 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
arr = (np.random.rand(10)*100).astype(np.float32)
print(arr)
integer_32_bit = arr.view(np.int32)
integer_32_bit[:] = arr
print(integer_32_bit)

# 54. How to read the following file?
from io import StringIO
fake_file = StringIO('''1, 2, 3, 4, 5
                        6,  ,  , 7, 8
                         ,  , 9, 10,11''')
Z = np.genfromtxt(fake_file, delimiter = ",", dtype = np.int)
print(Z)

# 55. What is the equivalent of enumerate for numpy arrays?
arr = np.arange(9).reshape(3,3)
print("Enumerate:")
for index, value in np.ndenumerate(arr):
    print(index, value)

print("Index:")
for index in np.ndindex(arr.shape):
    print(index, arr[index])

# 56. Generate a generic 2D Gaussian-like array
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
dist = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0 #Standard deviation = 1 and Mean = 0
G = np.exp(-( (dist - mu)**2 / ( 2.0 * sigma**2 ) ) ) # Exponential formula for normal distribution
print(G)

#import matplotlib
#import matplotlib.pyplot
#matplotlib.pyplot.plot(G)
#matplotlib.pyplot.savefig("Q56.png")

# 57. How to randomly place p elements in a 2D array?
size = 10
p = 3
matrix = np.zeros((size, size))
np.put(matrix, np.random.choice(range(size*size), p, replace=False),2)
print(matrix)

# 58. Subtract the mean of each row of a matrix
before = np.random.randint(5, size = (3, 3))
print("Before: \n", before)
after = before - before.mean(axis=1).reshape(-1, 1)
print("After: \n", after)

# 59. How to sort an array by the nth column?
Z = np.random.randint(0,10,(3,3))
print("Unsorted: \n", Z)
print("Sorted: \n", Z[Z[:,1].argsort()])

# 60. How to tell if a given 2D array has null columns?
Z = np.random.randint(0, 2, size = (3, 3))
print("Any null columns? \n", Z)
print((~Z.any(axis=0)).any())

# 61. Find the nearest value from a given value in an array
arr = np.random.uniform(0, 1, 10)
find = 0.5
m = arr.flat[np.abs(arr - find).argmin()]
print(m)

# 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?
A = np.arange(3).reshape(3, 1)
B = np.arange(3).reshape(1, 3)
total = np.nditer([A, B, None])
for x, y, z in total: z[...] = x + y
print(total.operands[2])

# 63. Create an array class that has a name attribute
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "array_10")
print(Z.name)
print(Z)

# 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?
arr = np.arange(3, 10)
print("Initial: ", arr)
ind = np.arange(len(arr))
print("Index:   ", ind)
np.add.at(arr, ind, 1)
print("Add 1:   ", arr)

# 65. How to accumulate elements of vector (X) to an array (F) based on an index list (I)?
X = [1, 2, 2, 5, 4]
I = [1, 6, 4, 2, 1]
F = np.bincount(I, weights = X)
print(F)

# 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
w, h = 16, 16
I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
print(np.unique(I))
print("Number of unique colours:", len(np.unique(I)))

# 67. Considering a four dimensions array, how to get sum over the last two axis at once?
A = np.random.randint(0, 2, (3, 4, 3, 4))
print(A)
sum = A.sum(axis = (2, 3))
print(sum)

# 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (Similar to #65)
D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# 69. How to get the diagonal of a dot product?
A = np.random.uniform(0, 1, (5, 5))
B = np.random.uniform(0, 1, (5, 5))
dot_prod = np.dot(A, B)
diagonal = np.diag(dot_prod)
print("Dot product = \n", dot_prod)
print("Diagonal = \n", diagonal)

# 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?
Z = np.arange(1, 6)
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[0::nz+1] = Z
print(Z0)

# 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)?
A = np.ones((5, 5, 3))
B = 2*np.ones((5,5))
print(A * B[:, :, None])

# 72. How to swap two rows of an array?
A = np.arange(25).reshape(5,5)
print("Before swapping: \n", A)
A[[0,1]] = A[[1,0]]
print("After swapping: \n", A)

# 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles
faces = np.random.randint(0, 100, (10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F)*3, 2)
F = np.sort(F, axis=1)
G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
G = np.unique(G)
print(G)

# 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?
C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)

# 75. How to compute averages using a sliding window over an array?
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

Z = np.arange(20)
print("Data = ", Z)
print("Moving average = ", moving_average(Z, n=3))

# 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)

# 77. How to negate a boolean, or to change the sign of a float inplace?
Z = np.random.randint(0, 2, 10)
print("Original: \n", Z)
np.logical_not(Z, out=Z) #flips zeroes and ones
print("Flipped: \n", Z)
np.negative(Z, out=Z) #add negative signs
print("Sign changed: \n", Z)

# 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])?
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[..., 0])*T[:, 0] + (P0[:, 1]-p[..., 1])*T[:, 1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    dist = np.sqrt((D**2).sum(axis=1))
    return dist

P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10, (10, 2))
p  = np.random.uniform(-10, 10, ( 1, 2))
dist = distance(P0, P1, p)
print("Distance: \n", dist)

# 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])?
P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10,(10, 2))
p = np.random.uniform(-10, 10, (10, 2))
print(np.array([distance(P0, P1, p_i) for p_i in p]))

# 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary)
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (3,3)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

# 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (Similar to #76)
Z = rolling(np.arange(1, 15), 4)
print(Z)

# 82. Compute a matrix rank
Z = np.random.uniform(0,1,(8,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print("Matrix rank =", rank)

# 83. How to find the most frequent value in an array?
Z = np.random.randint(1, 10, 100)
frequency = np.bincount(Z)
print("Frequency =", frequency)
print("Most frequent number =", frequency.argmax())

# 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix
Z = np.random.randint(0, 5, (10, 10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape = (i, j, n, n), strides = Z.strides + Z.strides)
print(Z)
print(C)

# 85. Create a 2D array subclass such that Z[i,j] == Z[j,i]
class Symmetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symmetric, self).__setitem__((i,j), value)
        super(Symmetric, self).__setitem__((j,i), value)

def symmetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symmetric)

S = symmetric(np.random.randint(0,10,(5,5)))
print(S)

# 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1))
p = 10
n = 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)?
Z = np.ones((16,16))
print("Before:\n", Z)
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print("After:\n", S)

# 88. How to implement the Game of Life using numpy arrays? (WHAT?)
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(10,10))
for i in range(100): Z = iterate(Z)
print(Z)

# 89. How to get the n largest values of an array
arr = np.arange(10000)
np.random.shuffle(arr)
n_largest = 5

sort = np.argsort(arr)
sorted_value = arr[sort]
largest_values = sorted_value[-5:]
print("n largest values = ", largest_values)

# 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item)
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T
    
    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]
    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))

# 91. How to create a record array from a regular array? (WHAT?)
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)

# 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods
Z = np.random.rand(100000)

print(np.power(Z, 3))
print(Z*Z*Z)
print(np.einsum('i,i,i->i', Z, Z, Z))

# 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (WHAT?)
A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))
C = A[..., np.newaxis, np.newaxis]
C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(A)
print(B)
print(rows)

# 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])
Z = np.random.randint(0,5,(10,3))
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)

# 95. Convert a vector of ints into a matrix binary representation
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128, 255], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))

# 96. Given a two dimensional array, how to extract unique rows?
Z = np.random.randint(0,2,(10,3))
print(Z)
uZ = np.unique(Z, axis=0)
print(uZ)

# 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function
A = np.random.randint(0,5,10)
B = np.random.randint(0,5,10)
print("A:", A)
print("B:", B)
print(np.einsum('i->', A))          # np.sum(A)
print(np.einsum('i,i->i', A, B))    # A * B
print(np.einsum('i,i', A, B))       # np.inner(A, B) otherwise known as dot product
print(np.einsum('i,j->ij', A, B))   # np.outer(A, B) otherwise known as cross product

# 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples?
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)

# 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n.
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 12
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n) #find the index of array X that sums up to n
print(X[M])

# 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means).
X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5]) #95% CI
print(confint)
