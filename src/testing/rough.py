import numpy as np, ClusteringAlgs.clustering as clustering, graphmodels.sampling as sampling

from tqdm import tqdm as tqdm

def cycle(k, noise=0.1):
      F = 0.5 * np.ones((k,k))
      for i in range(k):
            F[i, (i+1)%k] = 1-noise
            F[i, (i-1)%k] = noise
      return F

def rand_comp(k, noise=0.1, random_state=None):
	if random_state != None:
		rand.seed(random_state)
	import numpy.random as rand
	dirs = (rand.rand(k,k) < 0.5)
	F = np.triu(dirs * (1-noise) + (1-dirs) * noise,1)
	
	return F + np.tril(1-F.T,-1) + np.diag(0.5 * np.ones(k))


def ari(X,Y):
    a = np.array(list(map(len, X)))
    b = np.array(list(map(len, Y)))
    Xs = np.zeros((len(X), a.sum()))
    Ys = np.zeros((a.sum(), len(Y)))
    for i, clX in enumerate(X):
            Xs[i, :] = np.isin(np.arange(a.sum()), clX)
    for j, clY in enumerate(Y):
            Ys[:, j] = np.isin(np.arange(a.sum()), clY)
    n = np.matmul(Xs, Ys)
    from scipy.special import comb
    num = comb(n, 2).sum() - (comb(a,2).sum()*comb(b,2).sum())/comb(a.sum(), 2)
    den = 0.5 * (comb(a,2).sum() + comb(b, 2).sum()) - (comb(a,2).sum()*comb(b,2).sum())/comb(a.sum(), 2)
    return num/den

P = np.ones(5)/5
C = np.array(
	[[0,1,0,0,0],
	 [0,0,1,0,0],
	 [0,0,0,1,0],
	 [0,0,0,0,1],
	 [1,0,0,0,0]]
	)

A, comms= sampling.DSBM_PA(N=25000, k=5, P=P, C=C, c=100 ,random_state=28)
clustsd = clustering.disim(A, 5, 5)

# Ads, commsds= sampling.DSBM(n=5000,k=5, p=0.0005, q=0.0005, random_state=28, F=rand_comp(5, noise=0.05), Herm=False)
# clustsd = clustering.disim(A, 5, 5)

ari(clustsd, comms)
