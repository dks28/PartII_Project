import numpy as np
import numpy.linalg as la
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power as f_mat_pow
from sklearn.preprocessing import normalize as normalise

def cluster(A, k, ϵ):
	n, m = A.shape
	assert n == m
	assert (A != A.H).nnz == 0
	λ, g = la.eigsh(A)
	print('computed eigenpairs')
	g = np.matrix(g[:, np.abs(λ) > ϵ])
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
