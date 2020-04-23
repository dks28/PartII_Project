import numpy as np
import numpy.random as rand
from scipy.sparse import csr_matrix
from tqdm import tqdm as tqdm

def sample(k, n, p, q, F, random_state=None, Herm=False):
	if not (random_state == None):
		rand.seed(random_state)
	assert F.shape == (k, k) 
	assert k >= 2 
	assert abs(p-0.5) <= 0.5 
	assert abs(q-0.5) <= 0.5
	assert np.all(F + F.T == 1)
	N = k * n
	res = np.zeros((N,N), dtype=complex)
	for c in tqdm(range(k)):
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

def cycle(k, η):
	F = 0.5 * np.ones((k,k))
	for i in range(k):
		F[i, (i+1) % k] = 1 - η
		F[i, (i-1) % k] = η
	return F

def random_complete(k, η, random_state=None):
	if random_state != None:
		rand.seed(random_state)
	dirs = rand.rand(k,k) < 0.5
	tmp1 = dirs * (1 - η)
	tmp2 = (1 - dirs) * η
	F_upper = np.triu(tmp1 + tmp2, 1)
	diag = np.diag(0.5 * np.ones(k))
	F_lower = np.tril(1 - F_upper.T, -1)
	F = F_upper + diag + F_lower
	return F

def line(k, η):
	F = 0.5 * np.ones((k, k))
	for i in range(k):
		F[i, i+1:] = 1 - η
		F[i+1:, i] = η
	return F
