import graphmodels.DSBM as DSBM
import graphmodels.DSBM_PA as DSBM_PA
import graphmodels.convert as convert
import graphmodels.hermify as hermify
from ClusteringAlgs import disim, herm
import testing.evaluate as evaluate
import itertools
import numpy as np
import scipy.sparse
import json
def run_test(seed,
             N,
	     k,
	     p,
	     q,
	     η,
	     interpolation):
	n = N // k
	C1 = np.zeros((k,k))
	itr = 0
	while any(C1.sum(axis=1) == 0):
		F = DSBM.random_complete(k, η, random_state=seed ** itr)
		PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
		C2 = PA_kwargs['C']
		C1[:,:] = C2
		for pos in range(k):
			C1[pos, pos] = 0
			
	C1 = C1 / C1.sum(axis=1).reshape(-1,1)
	print(C2)
	print(C1)

	PA_kwargs['C'] = interpolation * C1 + (1-interpolation) * C2
	
	aris_DSBM_disim = []
	aris_DSBM_herm = []
	
	
	GPA, comms = DSBM_PA.sample(a=10, **PA_kwargs, Herm=False, random_state=seed)
	GPA_Herm = hermify.to_herm(GPA)
	
	clusters_disim = disim.cluster(GPA, k, k, mode='R')
	clusters_herm = herm.cluster(GPA_Herm, k, ϵ=-1, RW=True)

	ari_disim = evaluate.ari(comms, clusters_disim)
	ari_herm = evaluate.ari(comms, clusters_herm)
#	mv_disim = evaluate.misclustered_vertices(comms, clusters_disim)
#	mv_herm = evaluate.misclustered_vertices(comms, clusters_herm)

	results = {
		'interpolation' : interpolation,
		'ari_disim': ari_disim,
		'ari_herm' : ari_herm,
	}
	return results

N = 25000
k = 5
p = 0.0045
q = p
η = 0.0
seeds = [8128]

for seed in seeds:
	for inter in np.linspace(0,1, 25, endpoint=True):
		results = run_test(seed, N, k, p, q, η, inter)
		with open(f'RWHerm_NoiseReduction_seed{seed}_N{N}_k{k}_p{p}_interpolation{inter}.json', 'w') as f:
			json.dump(results, f)
