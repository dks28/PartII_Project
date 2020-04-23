import numpy as np
import scipy.sparse as sprs
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power as f_mat_pow
from sklearn.preprocessing import normalize as normalise
import numpy.random as rand

def cluster(A, k, ϵ=-1, RW=True, norm=None, τ_self_loops=1.0, ω=0, r=0):
	n, m = A.shape
	assert n == m
	assert (A != A.H).nnz == 0
	
	if not (norm == 'densify'):
		import scipy.sparse.linalg as la
		from scipy.sparse.linalg import eigsh as eigsh

	if norm == 'interpolation':
		B = r * (A != 0)
		κ = 1.0 / np.sqrt( r**2 + 1 )
		A = κ * (A + B)
	
	if norm == 'densify':
		A2 = np.zeros((n,n), dtype=complex)
		import scipy.linalg as la
		from scipy.linalg import eigh as eigh
		A2 = np.zeros((n,n))
		R = rand.rand(n,n)
		D = (R < 0.5).astype(complex) - (R >= 0.5).astype(complex)
		D = np.triu(D) - np.triu(D).T
		D *= ω*1j
		A2 += A
		A2 += D
		A = A2
	
	if RW:
		out_degrees = np.abs(np.imag(A)).sum(axis=1)
		D = sprs.spdiags(out_degrees.reshape((1,-1)), 0, n, n)
		if norm == "self-loops":
			τ_sl = np.mean(out_degrees) * τ_self_loops
			D = sprs.spdiags((out_degrees + τ_sl).reshape((1,-1)), 0, n, n)
		D_ = D.power(-0.5)
		A = D_ @ A @ D_

	if ϵ == None:
		edges = A.nnz 
		avg_degree = edges // n
		# The next line assumes that p=q, which is the case for all our experiments
		pn = avg_degree / k
		ϵ = 10 * np.sqrt( pn*np.log(pn) )
	λ, g = eigh(A, eigvals=(n-k, n-1)) if norm == 'densify' else eigsh(A, k=k)
	print('computed eigenpairs')
	if ϵ == -1:
		g = np.matrix(g[:, np.argsort(np.abs(λ))[-k:]])
	else:
		g = np.matrix(g[:, np.abs(λ) > ϵ])
	print(g.shape)
	Ptmp = g @ g.H
	# Numerical errors can mean that imaginary parts near 0 remain--but mathematically,
	# P is real!
	P = np.real(Ptmp)
	print('computed P')
	kmeans = KMeans(n_clusters=k).fit(P)
	print('fit kmeans')
	#now need to get the clusters, rather than the cluster assignments:
	assignments = kmeans.labels_
	clusters = [[]] * k
	for cl in range(k):
		clusters[cl] = np.arange(n)[assignments==cl]

	return clusters
