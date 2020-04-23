import numpy as np
import pandas as pd
from ClusteringAlgs import herm, disim
from testing import evaluate
from graphmodels import DSBM, DSBM_PA
from graphmodels import convert, hermify

def run_experiments(seed=0, noise=0, tag="", norm=None):
	print("Running NCyMG-experiments for seed", seed)
	k = 5
	n = 4000
	F = DSBM.cycle(k=k, η=noise)
	if norm == None:
		for p in [0.002, 0.004, 0.006, 0.008]:
			q = p
			PA_kwargs = convert.DSBM_to_PA(k = k, n=n, p=p, q=q, F=F)
			A, comms = DSBM.sample(n=n, k=k, p=p, q=q, F=F, random_state=seed, Herm=False)
			print("finished sampling dsbm. moving to clustering.")
			cls = disim.cluster(A=A, kz=k, ky=k, norm=norm)
			ari_ds = evaluate.ari(comms, cls)
			result1 = {
				'model': ['DSBM'],
				'p': [p],
				'noise': [noise],
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
				'p': [p],
				'noise': [noise],
				'algorithm': ['Herm'],
				'ari': [ari_herm]
			}
			df = pd.DataFrame(data=result2)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)
			print('finished herm-dsbm, movnig on to disim-pa')
			A, comms = DSBM_PA.sample(random_state=seed, a=PA_kwargs['c'], Herm=False, **PA_kwargs)
			cls = disim.cluster(A=A, kz=k, ky=k, norm=norm)
			ari = evaluate.ari(comms, cls)
			result3 = {
				'model': ['DSBM_PA'],
				'p': [p],
				'noise': [noise],
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
				'p': [p],
				'noise': [noise],
				'algorithm': ['Herm'],
				'ari': [ari]
			}
			df = pd.DataFrame(data=result4)
			df.to_csv(f"results/{tag}.csv", mode='a', header=False)			
			print(f"Concluded experiments for p={p}")
	pass
