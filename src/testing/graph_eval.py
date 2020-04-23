import graphmodels.DSBM as DSBM
import graphmodels.DSBM_PA as DSBM_PA
import graphmodels.convert as convert
import graphmodels.hermify as hermify
import itertools
import numpy as np
"""
def run_test(seed,
             N,
	     k,
	     p,
	     q,
	     η):
	n = N // k
	F = DSBM.circle(k, η)
#	C = DSBM_PA.tree(k = k, η = 0.0, inner_edges=0.15)
#	print(F.shape)
	PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
#	PA_kwargs['C'] = C
	
	aris_DSBM_disim = []
	aris_DSBM_herm = []
	
	
	GDSBM, comms_DSBM = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=False)
	print([len(l) for l in comms_DSBM])
	clustersr = disim.cluster(GDSBM, k, k, mode='R')
	clustersl = disim.cluster(GDSBM, k, k, mode='L')
	print([len(l) for l in clustersr],
	      [len(l) for l in clustersl])
	ari_disimr = evaluate.ari(comms_DSBM, clustersr)
	ari_disiml = evaluate.ari(comms_DSBM, clustersl)
	print('Disim_R', ari_disimr)
	print()
	GDSBM = hermify.to_herm(GDSBM)
	clusters = herm.cluster(GDSBM, k, ϵ=-1)
	ari_herm = evaluate.ari(comms_DSBM, clusters)
	print('Herm', ari_herm)
	print()

	results = dict([])
	results['Herm'] = ari_herm
	results['DiSim_R'] = ari_disimr
	results['DiSim_L'] = ari_disiml
	return results
"""
def spectral_gap():
	# circle metagraph
	noises = np.linspace(0,0.5, 15)
	ks = np.array([3,5,7])
	ps = np.array([0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055])
	seeds = np.array([6, 28, 496, 8128, 33550336])
	ns = [3000, 5000] # np.array([1000, 1500, 2000, 2500, 5000])
	for (η, k, n, p) in itertools.product(noises, ks, ns, ps):
		q = p
		F = DSBM.circle(k, η)
		print(F.shape)
		PA_kwargs = convert.DSBM_to_PA(n=n, k=k, p=p, q=q, F=F)
		for seed in seeds:
	#		GDSBM = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=True)
			GPA = DSBM_PA.sample(**PA_kwargs)
			print(GPA.shape)
