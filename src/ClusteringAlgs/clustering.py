import numpy as np
import scipy.sparse.linalg as la
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power as f_mat_pow
from sklearn.preprocessing import normalize as normalise

def herm(A, k, ϵ):
	n, m = A.shape
	assert n == m
	assert (A != A.H).nnz == 0
	λ, g = la.eigsh(A)
	print('computed eigenpairs')
	g = np.matrix(g[:, np.abs(λ) > ϵ])
	P = np.real(g @ g.H)
	print('computed P')
	kmeans = KMeans(n_clusters=k).fit(P)
	print('fit kmeans')
	#now need to get the clusters, rather than the cluster assignments:
	assignments = kmeans.labels_
	clusters = [[]] * k
	for cl in range(k):
		clusters[cl] = np.arange(n)[assignments==cl]

	return clusters
 
def disim(A, ky, kz, τ=None):
	import scipy.sparse
	mat = lambda m : scipy.sparse.csr_matrix(m)

	n, m = A.shape
	assert n==m
	
	K = min(ky, kz)
	
	out_degrees = A.sum(axis=1)
	in_degrees = A.sum(axis=0)
	print('computed degrees')
	τ = out_degrees.mean() if τ == None else τ
	Pτ = mat(np.diagflat(in_degrees + τ))
	print('computed P')
	Oτ = mat(np.diagflat(out_degrees + τ))
	print('computed O')
	tmp1 = Oτ.power(-0.5)
	print('tmp1')
	tmp2 = Pτ.power(-0.5)
	print('tmp2')
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
	kmeans = KMeans(n_clusters=K, n_jobs=-1).fit(XR_)

	assignments = kmeans.labels_
	print('done')
	clusters = [[]] * K
	for cl in range(K):
		clusters[cl] = np.arange(n)[assignments==cl]
	
	return clusters
