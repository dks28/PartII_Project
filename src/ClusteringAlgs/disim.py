import numpy as np
from tqdm import tqdm as tqdm
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power as f_mat_pow
from sklearn.preprocessing import normalize as normalise
import numpy.random as rand

def cluster(A, ky, kz, τ=None, mode='R', norm=None, τ_self_loops=1.0, ω=0):
	import scipy.sparse
	mat = lambda m : scipy.sparse.csr_matrix(m)

	n, m = A.shape
	assert n==m

	if norm == 'densify':
		import scipy.linalg as la
		from scipy.linalg import svd as svds
		A2 = np.zeros((n,n))
		R = rand.rand(n,n)
		D = (R < 0.5).astype(float)
		D = np.triu(D) + (1 - np.triu(D).T)
		D *= ω
		A2 = A + D
		A = A2
	else:
		import scipy.sparse.linalg as la
		from scipy.sparse.linalg import svds as svds
	
	K = min(ky, kz)
	
	out_degrees = A.sum(axis=1)
	in_degrees = A.sum(axis=0)
	print('computed degrees')

	if τ == None:
		τ = out_degrees.mean()
	if norm == None:
		τ = 1 # otherwise get division by zero when no in-degree...
	elif norm == 'self-loops' and τ == None:
		τ = τ_self_loops * out_degrees.mean()

	Pvals = in_degrees + τ
	Pτ = scipy.sparse.spdiags(Pvals, 0, n, n)
	print('computed P')
	Ovals = out_degrees + τ
	Oτ = scipy.sparse.spdiags(np.reshape(Ovals, (1,-1)), 0, n, n)
	print('computed O')
	tmp1 = Oτ.power(-0.5)
	print('generated tmp1')
	tmp2 = Pτ.power(-0.5)
	print('generated tmp2')
	L =  tmp1 @ A @ tmp2
	print('computed L')
	U, Σ, V = svds(L, k=K, which='LM') if not (norm == 'densify') else svds(L)
	print('found svd:', 'U:', U.shape, 'V:', V.shape)
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
		kmeans = KMeans(n_clusters=kz, n_jobs=-1).fit(XR_)
	if mode == 'X':
		kmeans = KMeans(n_clusters=K, n_jobs=-1).fit(X_)

	assignments = kmeans.labels_
	print('done')
	clusters = [[]] * K
	for cl in range(K):
		clusters[cl] = np.arange(n)[assignments==cl]
	
	return clusters
