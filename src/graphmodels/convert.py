import numpy as np
def DSBM_to_PA(k,n,p,q,F):
	N = k * n
	c = int(np.ceil( ( p * n  + q * (k-1) * n ) / 2 ))
	print('c',c)
	P = np.ones(k) / k
	# to calculate C, consider the following: 
	# P(edge starting in i goes to j) = 
	# N(edges from i to j) / N(edges starting in i)
	# Thus
	C = np.zeros((k,k))
	zero = False
	for i in range(k):
		total = (F[i, :].sum() - 0.5) * q + p/2
		if total == 0:
			zero = True
		C[i,i] = p / (2 * total)
		for j in range(k):
			if i == j:
				pass
			else:
				C[i,j] = q * F[i,j] / total
	if zero:
		return 'total=0'
	
	kwargs = { 'N':N, 
		   'c':c, 
		   'P':P, 
		   'k':k, 
		   'C':C }
	
	return kwargs

def PA_to_DSBM(k, N, P, C, c):
	print(P)
	print(C)
	n = N // k
	Ciis = np.diag(C)
	p = (Ciis * c) / (P * N - 1) 
	p = p.sum()
	q = ( (1 - Ciis) * c ) / ((1 - P) * N)
	q = q.sum()
	F = P * C / ( (P * C) + (P * C).T )
	F = np.nan_to_num(F, nan=0.5)

	kwargs = { 'n':n,
	           'p':p,
		   'q':q,
		   'F':F }
	
	return kwargs
