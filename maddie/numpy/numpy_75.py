#!/usr/bin/env python
#numpy exercise questions 51 - 75

import numpy as np

#51
#structured array representing position (x, y) and color (r, g, b)
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)

#52
#convert 32 bit array into an integer 32 bit in place
B = (np.random.rand(10)*100).astype(np.float32)
R = B.view(np.int32)
R[:] = B
print(R)

#53
#random vector with shape (100, 2) represents coordinates - find point by point distances
V = np.random.random((10, 2))
X,Y = np.atleast_2d(V[:, 0], V[:, 1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

#54
#generate fake file
s = StringIO('''1, 2, 3, 4, 5
                6,  ,  , 7, 8
                 ,  , 9, 10, 11
''')
#read from file
A = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(A)

#55
#equivalent of enumerate in numpy
M = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(M):
    print(index, value)
for index in np.ndindex(M.shape):
    print(index, M[index])

#56
#generate generic 2D Gaussian-like array
X, Y = np.meshgrid(np.linspace(-1,1,10)), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / (2.0 * sigma**2) ) )
print(G)

#57
#randomly place p elements in a 2D array
K = np.arange(10).reshape(5, 2)
print(K)

p = 3
np.put(L, np.random.choice(range(10), p, replace=False),1)
print(L)

#58
#Substract mean of each row of a matrix
N = np.random.rand(5, 5)
M = N - N.mean(axis=1, keepdims=True)
print(M)

#59
#sort an array by the nth column 
J = np.random.randint(0, 9,(3, 3))
print(J)
print(J[J[:,1].argsort()])

#60
#tell if 2D array has null columns
S = np.random.randint(0, 9,(3, 3))
print(S)
print((~S.any(axis=0)).any())

#61
#find nearest value from a given value in an array
U = np.random.uniform(0, 1, 10)
u = 0.2
n = U.flat[np.abs(U - u).argmin()]
print(n)

#62
#compute sum of 2 arrays with shape (1, 3) and (3, 1) using interator
A = np.arange(3).reshape(1, 3)
B = np.arange(3).reshape(3, 1)
I = np.nditer([A, B, None])
for m, n, o in I: o[...] = m + n
print(I.operands[2])

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

T = Array(np.arange(10), "upto_10")
print(T.name)

#64
#add 1 to each element indexed by a second vector
newArr = np.arange(10)
print(newArr)
ind = np.arange(len(newArr))
print(ind)
np.add.at(newArr, ind, 1)
print(newArr)

#65
#accumulate elements of a vector (X) to an array (F) based on an index list (I)
X = [1, 2, 3, 4, 5]
I = [1, 6, 4, 2, 3]
F = np.bincount(I, X)
print(F)

#66
#w,h,3 image of dtype=ubyte compute the number of unique colors
w,h = 10,10
E = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
Q = E[...,0]*256*256 + E[...,1]*256 + E[...,2]
n = len(np.unique(Q))
print(n)

#67
#4D array, sum over last two axis at once
F = np.random.randint(0, 5,(5, 6, 5, 6))
s = F.sum(axis=(-2, -1))
print(s)


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
A = np.random.randint(0, 5,(5, 5))
B = np.random.randint(0, 5,(5, 5))
print(np.diag(np.dot(A, B)))

#70
#vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value
G = np.array([1, 2, 3, 4, 5])
z = 3
z0 = np.zeros(len(G) + (len(G)-1)*(z))
z0[::z+1] = G
print(z0)

#71
#multiply array of dimensions (5, 5, 3) by an array with dimensions (5, 5)
u = np.random.randint(0, 10,(5, 5, 3))
print(u)
t = 2*np.ones((5, 5))
print(u * t[:, :, None])

#72
#swap two rows of an array
n = np.arange(25).reshape(5, 5)
print(n)
n[[0, 1]] = n[[1, 0]]
print(n)

#73
#find set of unique line segments composing all the triangles from set of 10 triplets describing 10 triangles (with shared vertices)
faces = np.random.randint(0, 100,(10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F, axis=1)
G = F.view( dtype=[('p0', F.dtype), ('p1', F.dtype)] )
G = np.unique(G)
print(G)

#74
#array C is a bincount, how to produce an array A such that np.bincount(A) == C
B = np.bincount([1, 1, 2, 3, 4, 4, 6])
R = np.repeat(np.arange(len(B)), B)
print(R)

#75
#compute averages using a sliding window over an array
def ave(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
c = np.arange(20)
print(ave(c, n=3))


#

