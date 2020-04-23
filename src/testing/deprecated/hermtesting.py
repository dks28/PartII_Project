import graphmodels.DSBM as DSBM
import graphmodels.DSBM_PA as DSBM_PA
import graphmodels.convert as convert
import testing.evaluate as evaluate
import ClusteringAlgs.disim as disim
import ClusteringAlgs.herm as herm
import numpy as np

seeds = [1, 6, 28, 42, 64, 128, 496, 8128, 200, 350, 1000, 5000, 6000, 7000, 8000, 9000, 10000]
def run():
	n = 4000
	k = 3
	N = k*n
	p = 0.01
	q = 0.01
	P = np.ones(k) / k
	F = DSBM.random_complete(k, η=0.0, random_state=28)
#	C = DSBM_PA.tree(k = k, η = 0.0, inner_edges=0.15)
	print(F.shape)
	PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
#	PA_kwargs['C'] = C
	
	aris_DSBM_disim = []
	aris_DSBM_herm = []
	
	for seed in seeds:
		GDSBM, comms_DSBM = DSBM_PA.sample(random_state=seed, Herm=False, a=10, **PA_kwargs)
		print([len(l) for l in comms_DSBM])
		clusters = disim.cluster(GDSBM, k, k, mode='R')
		print([len(l) for l in clusters])
		ari = evaluate.ari(comms_DSBM, clusters)
		aris_DSBM_disim.append(ari)
		print('Disim', ari)
		GDSBM, comms_DSBM = DSBM_PA.sample(random_state=seed, Herm=True, a=10, **PA_kwargs)
		clusters = herm.cluster(GDSBM, k, 0.1)
		ari = evaluate.ari(comms_DSBM, clusters)
		print('Herm', ari)
		aris_DSBM_herm.append(ari)

	print('Mean Disim', np.mean(aris_DSBM_disim))
	print('Mean Herm', np.mean(aris_DSBM_herm))
