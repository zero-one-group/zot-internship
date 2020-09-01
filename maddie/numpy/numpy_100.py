#76
#1D array Z build a 2D array whose 1st row is (Z[0], Z[1], Z[2]) and subsequent row is shifted by 1 (last row should be (Z[-3], Z[-2], Z[-1])
import numpy as np

from numpy.lib import stride_tricks

def shift(a, section):
    shape = (a.size - section + 1, section)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = shift(np.arange(9),3)
print(Z)

#77
#negate a boolean, or change the sign of a float inplace

random_array = np.random.randint(0, 1, 100)
np.logical_not(random_array, out=random_array)

random_array = np.random.uniform(-1.0, 1.0, 100)
np.negative(random_array, out=random_array)

#78
#with 2 sets of points P0, P1 describing lines (2d) and a point p, how to compute distance from p to each line i (PO[i], P1[i])

def distance(P0, P1, p):
    difference = P1 - P0
    length = (difference**2).sum(axis=1)
    unit = -((P0[:,0]-p[...,0])*difference[:,0] + (P0[:,1]-p[...,1]*difference[:,1]) / length)
    unit = unit.reshape(len(unit),1)
    distance_result = P0 + unit*difference - p 
    return np.sqrt((distance_result**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10,10,(1, 2))
print(distance(P0, P1, p))

#79
#2 points P0, P1 describing lines and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i], P1[i])

P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
#based on previous function
print(np.array([dis(P0,P1,p_i) for p_i in p]))

#80
#use an arbitrary array to write a function that extracts a subpart with a fixed shape and centered on a given element

arbitrary_array = np.random.randint(0, 9,(3, 3))
shape = (5, 5)
fill = 0
position = (1, 1)

ones_array = np.ones(shape, dtype=arbitrary_array.dtype)*fill
list_position = np.array(list(position)).astype(int)
ones_shape = np.array(list(ones_array.shape)).astype(int)
arbitrary_shape = np.array(list(arbitrary_array.shape)).astype(int)

ones_start = np.zeros((len(shape),)).astype(int)
ones_stop = np.array(list(shape)).astype(int)
arbitrary_start = (list_position-ones_shape//2)
arbitrary_stop = (list_position+ones_shape//2)+arbitrary_shape%2


slice_ones = [slice(start, stop) for start,stop in zip(ones_start,ones_stop)]
slice_arbitrary = [slice(start,stop) for start,stop in zip(arbitrary_start,arbitrary_stop)]
R[slice_ones] = A[slice_arbitrary]
print(arbitrary_array)
print(ones_array)

#81
#consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6],...,[11,12,13,14]]

from numpy.lib import stride_tricks
Z = np.arange(1, 15, dtype=np.uint32)
R = stride_tricks.as_strided(A, (11, 4), (4, 4))
print(R)

#82
#compute a matrix rank

new_matrix = np.random.uniform(0, 1, (10, 10))
B, C, D = np.linalg.svd(new_matrix)
rank = np.sum(new_matrix > 1e-10)
print(rank)

#83
#find the most frequent value in an array

array = np.random.randint(0, 10, 50)
print(np.bincount(array).argmax())

#84
#Extract all the contiguous 3x3 blocks from a random 10x10 matrix

from numpy.lib import stride_tricks

random_matrix = np.random.randint(0, 5,(10, 10))
m = 3
h = 1 + (random_matrix.shape[0]-3)
k = 1 + (random_matrix.shape[1]-3)
blocks = stride_tricks.as_strided(random_matrix, shape=(h, k, m, m), strides=random_matrix.strides + random_matrix.strides)
print(blocks)

#85
#create 2D array subclass such that Z[i,j] == Z[j,i]

class mirror (np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(mirror, self).__setitem__((i,j), value)
        super(mirror, self).__setitem__((j,i), value)
def mirror(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(mirror)
S = mirror(np.random.randint(0, 10,(5, 5)))
S[2, 3] = 42
print(S)

#86
#use a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1) to compute the sum of the p matrix products at once (result has shape (n,1))

p, n = 10, 20
N = np.ones((p,n,n))
T = np.ones((p,n,1))
Q = np.tensordot(N, T, axes=[[0, 2]], [0, 1])
print(Q)

#87
#use a 16x16 array to get the block sum (block size is 4x4)
array = np.ones((16, 16))
block_size = 4
block_sum = np.add.reduceat(np.add.reduceat(A, np.arange(0, A.shape[0], L), axis=0),
                                       np.arange(0, A.shape[1], L), axis=1)
print(block_sum)

#88
#implement the game of life using numpy arrays

def iterate(Z):
    #count surrounding neighbours
    neighbours = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
    #The rules
    birth = (neighbours==3) & (Z[1:-1,1:-1]==0)
    survive = ((neighbours==2) | (neighbours==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    # Apply rules
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)

#89
#get the n largest values of an array

array = np.arange(10000)
np.random.shuffle(array)
n = 3
print(array[np.argsort(array)[-n:]])

#90
#give arbitrary number of vectors, build the cartesian product(every combination of every item)

def cartesian_product(array):
    array = [np.asarr(a) for a in array]
    shape = (len(x) for x in arr)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arr), -1).T

    for n, array in enumerate(array):
        ix[:, n] = arr[n][ix[:, n]]
        return ix

print(cartesian_product(([1, 2, 3], [4, 5], [6, 7])))

#91
#record array from regular array

regular_array = np.array([("Testing", 2.5, 3),
              ("record", 3.6, 2)])
record_array = np.core.records.fromarrays(regular_array.T,
                               names = 'col1, col2, col3',
                               formats = 'S8, f8, i8')
print(record_array)

#92
#large vector Z, compute Z to the power of 3 using 3 different methods

Z = np.random.randint(0, 10(5e6))
np.power(Z, 3)
print(Z*Z*Z)
np.einsum('i, i, i->i', Z, Z, Z)

#93
#using two arrays A and B of shape (8,3) and (2,2) how to find rows of A that contain elements of each row of B regardless of the order of elements in B

X = np.random.randint(0,5,(8,3))
Y = np.random.randint(0,5,(2,2))

Z = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(Z.any((3,1)).all(1))[0]
print(rows)

#94
#10x3 matrix, extract rows with unequal values
matrix = np.random.randint(0, 4,(10, 3))
print(matrix)
all_matrix = np.all(matrix[:,1:] == matrix[:,:-1], axis=1)
extract = matrix[~all_matrix]
print(extract)

#95
#vector of ints into a matrix binary representation
ints_vector = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
binary_rep = ((ints_vector.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(binary_rep[:,::-1])

ints_vector = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(ints[:, np.newaxis], axis=1))

#96
#from  a 2D array how to extract unique rows?
two_d_array = np.random.randint(0, 10,(5, 2))
print(np.unique(two_d_array, axis=0))

#97
#with 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function
A = np.random.uniform(0, 1, 10)
B = np.random.uniform(0, 1, 10)

np.einsum('i->', A)
np.einsum('i, i->i', A, B)
np.einsum('i, i', A, B)
np.einsum('i, j->ij', A, B)

#98
#consider a path described by two vectors (X, Y) how to sample it using equidistant samples?
O = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*O*np.cos(O)
y = a*O*np.sin(O)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)
r_int = np.linspace(0, r.max(), 200)
x_int = np.interp(r_int, r, x)
y_int = np.interp(r_int, r, y)

#99
#given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n

X= np.asarray([[1.0, 0.0, 3.0, 8.0],
               [2.0, 0.0, 1.0, 1.0],
               [1.5, 2.5, 1.0, 0.0]])

n = 4
multinomial_distribution = np.logical_and.reduce(np.mod(X, 1) == 0, axis = -1)
mutinomial_distribution &= (X.sum(axis=-1) == n)
print(X[multinomial_distribution])

#100
#compute bootstrapped 95% CIs for the mean of a 1D array X 
X = np.random.randn(100)
number = 1000
index = np.random.randint(0, X.size, (number, X.size))
means = X[index].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
