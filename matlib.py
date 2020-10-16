"""
matlib.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""
import numpy as np
import scipy as sp
import scipy.linalg as la
from numba import njit

# Problem 0

# Part A
def solve_chol(A, b):
    """
    solve A * x = b for x

    assume A is SPD

    L @ L.T x = b
    x = L^{-T} L^{-1}  b
    """

    L = la.cholesky(A, lower=True)
    x = la.solve_triangular(L, b, lower=True)
    return la.solve_triangular(L, x, trans='T', lower=True)

# part B
def matrix_pow(A, n):

	lam, V = la.eigh(A)
	lamn = lam**n
	return V @ np.diag(lamn) @ V.T


# part D
def abs_det(A):

    P, L, U = la.lu(A)
    Ud = np.diag(U) # L and P have det 1
    return abs(np.prod(Ud))


# Problem 1

# part A
@njit
def matmul_ijk(B, C):
	p, r = B.shape
	s, q = C.shape
	if r != s:
		raise AssertionError("Incompatible Shapes!")

	A = np.zeros((p, q))
	for i in range(p):
		for j in range(q):
			for k in range(r):
				A[i,j] = A[i,j] + B[i,k] * C[k,j]

	return A

@njit
def matmul_ikj(B, C):
	p, r = B.shape
	s, q = C.shape
	if r != s:
		raise AssertionError("Incompatible Shapes!")

	A = np.zeros((p, q))
	for i in range(p):
		for k in range(r):
			for j in range(q):
				A[i,j] = A[i,j] + B[i,k] * C[k,j]

	return A

@njit
def matmul_jik(B, C):
	p, r = B.shape
	s, q = C.shape
	if r != s:
		raise AssertionError("Incompatible Shapes!")

	A = np.zeros((p, q))
	for j in range(q):
		for i in range(p):
			for k in range(r):
				A[i,j] = A[i,j] + B[i,k] * C[k,j]

	return A

@njit
def matmul_jki(B, C):
	p, r = B.shape
	s, q = C.shape
	if r != s:
		raise AssertionError("Incompatible Shapes!")

	A = np.zeros((p, q))
	for j in range(q):
		for k in range(r):
			for i in range(p):
				A[i,j] = A[i,j] + B[i,k] * C[k,j]

	return A

@njit
def matmul_kij(B, C):
	p, r = B.shape
	s, q = C.shape
	if r != s:
		raise AssertionError("Incompatible Shapes!")

	A = np.zeros((p, q))
	for k in range(r):
		for i in range(p):
			for j in range(q):
				A[i,j] = A[i,j] + B[i,k] * C[k,j]

	return A

@njit
def matmul_kji(B, C):
	p, r = B.shape
	s, q = C.shape
	if r != s:
		raise AssertionError("Incompatible Shapes!")

	A = np.zeros((p, q))
	for k in range(r):
		for j in range(q):
			for i in range(p):
				A[i,j] = A[i,j] + B[i,k] * C[k,j]

	return A

# Part B
@njit
def matmul_blocked(B, C):
	p, r = B.shape
	s, q = C.shape

	if not (p == r == s == q):
		raise AssertionError("Only square matrices supported")

	n = p

	if n < 65:
		return matmul_ikj(B, C)

	A = np.zeros((n, n))

	slices = (slice(0, n//2), slice(n//2, n))
	for I in slices:
		for K in slices:
			for J in slices:
				A[I, J] = A[I, J] + matmul_blocked(B[I, K], C[K, J])

	return A


# Part C
@njit
def matmul_strassen(B, C):
	p, r = B.shape
	s, q = C.shape

	if not (p == r == s == q):
		raise AssertionError("Only square matrices supported")

	n = p

	if n < 65:
		return matmul_ikj(B, C)

	A = np.zeros((n, n))

	s1 = slice(0, n//2)
	s2 = slice(n//2, n)

	B11, B12, B21, B22 = B[s1,s1], B[s1,s2], B[s2, s1], B[s2, s2]
	C11, C12, C21, C22 = C[s1,s1], C[s1,s2], C[s2, s1], C[s2, s2]

	M1 = matmul_strassen((B11 + B22), (C11 + C22))
	M2 = matmul_strassen((B21 + B22), C11)
	M3 = matmul_strassen(B11, (C12 - C22))
	M4 = matmul_strassen(B22, (C21 - C11))
	M5 = matmul_strassen((B11 + B12), C22)
	M6 = matmul_strassen((B21 - B11), (C11 + C12))
	M7 = matmul_strassen((B12 - B22), (C21 + C22))

	A[s1, s1] = M1 + M4 - M5 + M7
	A[s1, s2] = M3 + M5
	A[s2, s1] = M2 + M4
	A[s2, s2] = M1 - M2 + M3 + M6

	return A


# Problem 2
def markov_matrix(n):
	"""
	Matrix for random walk in 1-dimension on n states
	"""
	A = np.zeros((n,n))
	# internal transitions
	for j in range(1,n-1):
		A[j-1,j] = 0.5
		A[j+1,j] = 0.5

	# boundaries
	A[1,0] = 0.5; A[0,0] = 0.5
	A[n-2,n-1] = 0.5; A[-1,-1] = 0.5
	return A
