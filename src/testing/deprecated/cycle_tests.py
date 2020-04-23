import graphmodels.DSBM as DSBM
import graphmodels.DSBM_PA as DSBM_PA
import graphmodels.convert as convert
import graphmodels.hermify as hermify
import testing.evaluate as evaluate
import ClusteringAlgs.disim as disim
import ClusteringAlgs.herm as herm
import numpy as np

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
	GPA, comms_pa = DSBM_PA.sample(a=10, **PA_kwargs, random_state=seed)
	print([len(l) for l in comms_DSBM])
	clustersds_dsbm = disim.cluster(GDSBM, k, k, mode='R')
	clustershm_dsbm =  herm.cluster(hermify.to_herm(GDSBM), k, k)
	clustersds_pa = disim.cluster(GPA, k, k, mode='R')
	clustershm_pa =  herm.cluster(hermify.to_herm(GPA), k, k)
	ari_dsbm_disim = evaluate.ari(comms_DSBM, clustersds_dsbm)
	ari_dsbm_herm  = evaluate.ari(comms_DSBM, clustershm_dsbm)
	ari_pa_disim = evaluate.ari(comms_pa, clustersds_pa)
	ari_pa_herm  = evaluate.ari(comms_pa, clustershm_pa)
	miscl_vs_disim_dsbm = evaluate.misclustered_vertices(comms_DSBM, clustersds_dsbm)
	miscl_vs_herm_dsbm  = evaluate.misclustered_vertices(comms_DSBM, clustershm_dsbm)
	miscl_vs_disim_pa = evaluate.misclustered_vertices(comms_pa, clustersds_pa)
	miscl_vs_herm_pa  = evaluate.misclustered_vertices(comms_pa, clustershm_pa)

	print()

	results = dict([])
	results['DiSim_DSBM_M'] = miscl_vs_disim_dsbm
	results['DiSim_PA_M'] = miscl_vs_disim_pa
	results['Herm_DSBM_M'] = miscl_vs_herm_dsbm
	results['Herm_PA_M'] = miscl_vs_herm_pa
	results['DiSim_DSBM_A'] = ari_dsbm_disim
	results['DiSim_PA_A'] = ari_pa_disim
	results['Herm_DSBM_A'] = ari_dsbm_herm
	results['Herm_PA_A'] = ari_pa_herm
	return results
