import numpy as np
import numpy.random as rand
from scipy.sparse import csr_matrix

def sample(k, n, p, q, F, random_state=None, Herm=True):
	if not (random_state == None):
		rand.seed(random_state)
	assert (F.shape == k, k) 
	assert k >= 2 
	assert abs(p-0.5) <= 0.5 
	assert abs(q-0.5) <= 0.5
	assert np.all(F + F.T == 1)
	N = k * n
	res = np.zeros((N,N), dtype=complex)
	for c in range(k):
		for d in range(c,k):
			prob = p if c == d else q
			cd_ns = rand.rand(n,n) <= prob
			dirs = 2 * (rand.rand(n,n) <= F[c,d]) - 1
			cd_ns = cd_ns * dirs
			res[c * n : (c+1) * n, d * n : (d+1) * n] = cd_ns
			
	if Herm:
		res = (np.triu(res) - np.triu(res).T) * 1j
	else:
		res = (np.triu(res) - np.triu(res).T) > 0
	
	# Need to make matrix sparse format to prevent memory errors
	res = csr_matrix(res)
	# Need to return both adjacency matrix and the underlying clusters.
	clusters = np.reshape(np.arange(N), (k,n))
	return res, clusters
