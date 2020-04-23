import numpy as np
import pandas as pd
from ClusteringAlgs import herm, disim
from testing import evaluate
from graphmodels import DSBM, DSBM_PA
from graphmodels import convert, hermify

def run_experiments(seed=0, noise=0, tag="", norm=None):
	η = noise
	print("Running ROCMG-experiments for seed", seed)
	k = 5
	n = 4000
	its = 0
	p = 0
	q = 0.0045
	dic = 'total=0'
	while dic == 'total=0':
		F = DSBM.random_complete(k=k, η=0, random_state=seed + its)
		dic = convert.DSBM_to_PA(k = k, n=n, p=p, q=q, F=F)
		its += 1
	dic['C'] = η * np.diag(np.ones(k)) + (1-η) * dic['C']
	dic2 = convert.PA_to_DSBM(**dic)
	if norm == None:
		A, comms = DSBM.sample(n=n, k=k, p=dic2['p'], q=dic2['q'], F=F, random_state=seed, Herm=False)
		print("finished sampling dsbm. moving to clustering.")
		cls = disim.cluster(A=A, kz=k, ky=k, norm=norm)
		ari_ds = evaluate.ari(comms, cls)
		result1 = {
			'model': ['DSBM'],
			'average_edge_probability': [q],
			'percentage_intra_edges': [η],
			'algorithm': ['DiSim'],
			'ari': [ari_ds]
		}
		df = pd.DataFrame(data=result1)
		df.to_csv(f'results/{tag}.csv', mode='a', header=False)
		print('finished disim-dsbm, moving on to herm-dsbm')
		A = hermify.to_herm(A)
		cls = herm.cluster(A=A, k=k, norm=norm)
		ari_herm = evaluate.ari(comms, cls)
		result2 = {
			'model': ['DSBM'],
			'average_edge_probability': [q],
			'percentage_intra_edges': [η],
			'algorithm': ['Herm'],
			'ari': [ari_herm]
		}
		df = pd.DataFrame(data=result2)
		df.to_csv(f"results/{tag}.csv", mode='a', header=False)
		print('finished herm-dsbm, movnig on to disim-pa')
		A, comms = DSBM_PA.sample(random_state=seed, a=dic['c'], Herm=False, **dic)
		cls = disim.cluster(A=A, kz=k, ky=k, norm=norm)
		ari = evaluate.ari(comms, cls)
		result3 = {
			'model': ['DSBM_PA'],
			'average_edge_probability': [q],
			'percentage_intra_edges': [η],
			'algorithm': ['DiSim'],
			'ari': [ari]
		}
		print("finished disim-pa, moving on to herm-pa")
		df = pd.DataFrame(data=result3)
		df.to_csv(f"results/{tag}.csv", mode='a', header=False)
		A = hermify.to_herm(A)
		cls = herm.cluster(A=A, k=k, norm=norm)
		ari = evaluate.ari(comms, cls)
		result4 = {
			'model': ['DSBM_PA'],
			'average_edge_probability': [q],
			'percentage_intra_edges': [η],
			'algorithm': ['Herm'],
			'ari': [ari]
		}
		df = pd.DataFrame(data=result4)
		df.to_csv(f"results/{tag}.csv", mode='a', header=False)			
		print(f"Concluded experiments for p={p}")
	elif norm == 'interpolation':
		A, comms = DSBM.sample(n=n, k=k, p=dic2['p'], q=dic2['q'], F=F, random_state=seed, Herm=False)
		print("finished sampling dsbm. moving to clustering.")
		A = hermify.to_herm(A)
		for r in np.arange(1,11):
			cls = herm.cluster(A=A, k=k, norm=norm, r=r)
			ari_herm = evaluate.ari(comms, cls)
			result2 = {
				'model': ['DSBM'],
				'average_edge_probability': [q],
				'percentage_intra_edges': [η],
				'r': [r],
				'ari': [ari_herm]
			}
			df = pd.DataFrame(data=result2)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)
		print('finished herm-dsbm, movnig on to disim-pa')
		A, comms = DSBM_PA.sample(random_state=seed, a=dic['c'], Herm=False, **dic)
		A = hermify.to_herm(A)
		for r in np.arange(1,11):
			cls = herm.cluster(A=A, k=k, norm=norm)
			ari = evaluate.ari(comms, cls)
			result4 = {
				'model': ['DSBM_PA'],
				'average_edge_probability': [q],
				'percentage_intra_edges': [η],
				'r': [r],
				'ari': [ari]
			}
			df = pd.DataFrame(data=result4)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)			
		print(f"Concluded experiments for p={p}")
