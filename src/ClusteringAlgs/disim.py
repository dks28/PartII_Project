import numpy as np
import scipy.sparse.linalg as la
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power as f_mat_pow
from sklearn.preprocessing import normalize as normalise

def cluster(A, ky, kz, τ=None, mode='L'):
	import scipy.sparse
	mat = lambda m : scipy.sparse.csr_matrix(m)

	n, m = A.shape
	assert n==m
	
	K = min(ky, kz)
	
	out_degrees = A.sum(axis=1)
	in_degrees = A.sum(axis=0)
	print('computed degrees')
	if τ == None:
		τ = out_degrees.mean()
	Pvals = in_degrees + τ
	Pτ = scipy.sparse.spdiags(Pvals, 0, n, n)
	print('computed P')
	Ovals = out_degrees + τ
	Oτ = scipy.sparse.spdiags(Ovals, 0, n, n)
	print('computed O')
	tmp1 = Oτ.power(-0.5)
	print('generated tmp1')
	tmp2 = Pτ.power(-0.5)
	print('generated tmp2')
	L =  tmp1 @ A @ tmp2
	print('computed L')
	U, Σ, V = la.svds(L)
	print('found svd')
	XL = U[:, np.argsort(Σ)[-K:]]
	XR = V.T[:, np.argsort(Σ)[-K:]]
	XL_, XR_ = normalise(XL), normalise(XR)
	X_ = np.zeros((2*n, K))
	X_[:n, :] = XL_
	X_[n:, :] = XR_
	print('about to cluster')
	kmeans = None
	if mode == 'L':
		kmeans = KMeans(n_clusters=ky, n_jobs=-1).fit(XL_)
	if mode == 'R':
		kmeans = KMeans(n_clusters=kx, n_jobs=-1).fit(XL_)
	if mode == 'X':
		kmeans = KMeans(n_clusters=K, n_jobs=-1).fit(X_)

	assignments = kmeans.labels_
	print('done')
	clusters = [[]] * K
	for cl in range(K):
		clusters[cl] = np.arange(n)[assignments==cl]
	
	return clusters
