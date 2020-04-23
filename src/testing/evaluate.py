import numpy as np
from scipy.special import comb

def ari(X,Y):
	kX = len(X)
	kY = len(Y)
	a = np.array(list(map(len, X)))
	b = np.array(list(map(len, Y)))
	n = a.sum()
	assert b.sum() == n
	Xs = np.zeros((kX, n))
	Ys = np.zeros((n, kY))
	for i, clX in enumerate(X):
		elems = np.arange(n)
		Xs[i, :] = np.isin(elems, clX)
	for j, clY in enumerate(Y):
		elems = np.arange(n)
		Ys[:, j] = np.isin(elems, clY)
	N = np.matmul(Xs, Ys)
	num1 = comb(N, 2).sum()
	num2 = comb(a, 2).sum() * comb(b, 2).sum() / comb(n, 2)
	den1 = 0.5 * ( comb(a, 2).sum() + comb(b, 2).sum() ) 
	den2 = ( comb(a,2).sum() * comb(b,2).sum() ) / comb(n, 2)
	return (num1 - num2) / (den1 - den2)

def misclustered_vertices(X, Y):
	kX = len(X)
	kY = len(Y)
	assert kX == kY
	k = kX
	k_ = np.arange(k)
	from itertools import permutations
	from itertools import chain
	min_misclass = float('inf')
	from tqdm import tqdm as tqdm
	print('checking permutations for misclustered vertices')
	for perm in tqdm(permutations(k_)):
		miscl_verts = sum(sum([[[x for x in X[c] if not(x in Y[perm[c]])], [y for y in Y[perm[c]] if not(y in X[c])]] for c in k_], []), [])
		miscl_verts = len(np.unique(miscl_verts))
		min_misclass = min(min_misclass, miscl_verts)
	return min_misclass
