import scipy.sparse
import numpy as np

def to_herm(A):
	A_ = scipy.sparse.csr_matrix(A, dtype=complex)
	A_ = A_ * 1j
	A_ = A_ - A_.T
	return A_

def from_herm(A):
	A_ = np.imag(A)
	A_ = np.where(A_ > 0, A_, 0)
	A_ = scipy.sparse.csr_matrix(A_, dtype=int)
	return A_
